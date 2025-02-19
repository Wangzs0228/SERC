import sys
sys.path.append("./")
import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from thop import profile

from configs import CFG
from metric import Metric
from models import build_model
from criterions import build_criterion
from optimizers import build_optimizer
from schedulers import build_scheduler
from datas import build_dataset, build_dataloader, build_iterator
from datas import visualize
from datas import convert_to_color,display_predictions,count_sliding_window,sliding_window,grouper
import wandb

def test(net,net1,net2, img, patch_size,batch_size,n_classes,step=1):
    """
    Test a model on a specific image
    """
    net.eval()
    center_pixel = True
    device ="cuda"
    kwargs = {
        "step": step,
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
            indices = [b[1:] for b in batch]
            data = data.to(device)
            # f,output = net(data)
            fea = net(data)
            p1_t, p2_t = net1(fea)[-1], net2(fea)[-1]
            output = (p1_t + p2_t) / 2

            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs
def extract(net,net1,net2, data_loader, device):
    net.eval()
    net1.eval()
    net2.eval()
    return_f = []
    return_y = []
    return_label = []

    for i, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            fea = net(data)
            p1_t, p2_t = net1(fea)[-1], net2(fea)[-1]
            y_t = (p1_t + p2_t) / 2
            return_f.append(fea[:,:,0,0].detach())
            return_y.append(y_t.detach())
            return_label.append(target.detach())

    r_f = torch.cat(return_f)
    r_y = torch.cat(return_y)
    r_l = torch.cat(return_label)

    if r_l.min()==0:
        r_l = r_l +1 
    return r_f, r_y ,r_l

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('../../runs', datetime.now().strftime('%Y%m%d-%H%M%S-train')),
                        help='path for experiment output files')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to validate in the training process')
    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=1,
                        help='number of nodes / machines')
    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=1,
                        help='number of GPUs per node / machine')
    parser.add_argument('-r',
                        '--rank-node',
                        type=int,
                        default=0,
                        help='ranking of the current node / machine')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='backend for PyTorch DDP')
    parser.add_argument('--master-ip',
                        type=str,
                        default='localhost',
                        help='network IP of the master node / machine')
    parser.add_argument('--master-port',
                        type=str,
                        default='8888',
                        help='network port of the master process on the master node / machine')
    parser.add_argument('--seed',
                        type=int,
                        default=30,
                        help='random seed')

    args = parser.parse_args()
    # number of GPUs totally, which equals to the number of processes
    args.path = os.path.join(args.path, str(args.seed))
    args.world_size = args.nodes * args.gpus
    return args


def worker(rank_gpu, args):
    # create experiment output path if not exists
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)
    wandb.init(project="HSI_DOMAIN_ADAPTATION", name=args.path[7:-2])
    wandb.config.update(CFG, allow_val_change=True)
    
    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())
    # print(CFG)
    assert CFG.EPOCHS % args.world_size == 0, 'cannot apportion epoch to gpus averagely'
    # log to file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'train.log')),
            logging.StreamHandler(),
        ])

    # rank of global worker
    rank_process = args.gpus * args.rank_node + rank_gpu
    dist.init_process_group(backend=args.backend,
                            init_method=f'tcp://{args.master_ip}:{args.master_port}',
                            world_size=args.world_size,
                            rank=rank_process)
    # number of workers
    logging.info('train on {} of {} processes'.format(rank_process + 1, dist.get_world_size()))

    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # initialize TensorBoard summary writer
    # rank of global worker
    if dist.get_rank() == 0:
        writer = SummaryWriter(logdir=args.path)

    # build dataset
    source_dataset = build_dataset('train')
    target_dataset = build_dataset('test')
    val_dataset = target_dataset
    assert source_dataset.num_classes == val_dataset.num_classes
    logging.info(
        "Number of train {}, val {}, test {}".format(len(source_dataset), len(val_dataset), len(target_dataset)))
    NUM_CHANNELS = source_dataset.num_channels
    NUM_CLASSES = source_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    # build data sampler
    source_sampler = DistributedSampler(source_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    target_sampler = DistributedSampler(target_dataset, shuffle=True)
    # build data loader
    source_dataloader = build_dataloader(source_dataset, sampler=source_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)
    target_dataloader = build_dataloader(target_dataset, sampler=target_sampler)
    # build data iteration
    source_iterator = build_iterator(source_dataloader)
    target_iterator = build_iterator(target_dataloader)

    # build model
    FE, C1, C2 = build_model(NUM_CHANNELS, NUM_CLASSES)
    FE.to(device)
    C1.to(device)
    C2.to(device)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    dis_criterion = build_criterion(loss_names[1])
    dis_criterion.to(device)
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_fe = build_optimizer(FE)
    optimizer_c1 = build_optimizer(C1)
    optimizer_c2 = build_optimizer(C2)
    # build scheduler
    scheduler1 = build_scheduler(optimizer_fe)
    scheduler2 = build_scheduler(optimizer_c1)
    scheduler3 = build_scheduler(optimizer_c2)
    # grad scaler
    scaler = GradScaler()
    # DDP
    FE = DistributedDataParallel(FE)
    C1 = DistributedDataParallel(C1)
    C2 = DistributedDataParallel(C2)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_PA = 0.
   
    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        FE.load_state_dict(checkpoint['FE']['state_dict'])
        C1.load_state_dict(checkpoint['C1']['state_dict'])
        C2.load_state_dict(checkpoint['C2']['state_dict'])
        optimizer_fe.load_state_dict(checkpoint['optimizer_fe']['state_dict'])
        optimizer_c1.load_state_dict(checkpoint['optimizer_c1']['state_dict'])
        optimizer_c2.load_state_dict(checkpoint['optimizer_c2']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_PA = checkpoint['metric']['PA']
        best_epoch = checkpoint['optimizer']['best_epoch']
        logging.info('load checkpoint {} with PA={:.4f}, epoch={}'.format(args.checkpoint, best_PA, epoch))

    # train - validation loop

    while True:
        epoch += 1
        # apportion epochs to each gpu averagely
        if epoch > int(CFG.EPOCHS / args.world_size):
            logging.info("Best epoch:{}, PA:{:.3f}".format(best_epoch, best_PA))
            wandb.log({"Best epoch":best_epoch,"PA":best_PA})

            if dist.get_rank() == 0:
                writer.close()
                if CFG.DATASET.NAME!='ShangHang':
                    f_s,y_s,label_s = extract(FE,C1,C2, source_dataloader, "cuda",)
                    f_t,y_t,label_t = extract(FE,C1,C2, val_dataloader, "cuda",)
                    # tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
                    num = np.minimum(len(f_s),len(f_t))
                    if (num>1000):
                        num=1000
                    visualize(f_s.cpu().detach()[0:num], f_t.cpu().detach()[0:num],y=torch.cat((y_s[0:num],y_t[0:num]),0),label=list(range(1,NUM_CLASSES+1)),name="mcd")
                    visualize(f_s.cpu().detach()[0:num], f_t.cpu().detach()[0:num],y=torch.cat((label_s[0:num],label_t[0:num]),0),label=list(range(1,NUM_CLASSES+1)),name="mcd")

                probabilities = test(FE,C1,C2, target_dataset.data.permute(1,2,0), patch_size=CFG.DATASET.PATCH.HEIGHT ,batch_size=CFG.DATALOADER.BATCH_SIZE,n_classes=NUM_CLASSES,)
                prediction = np.argmax(probabilities, axis=-1)[CFG.DATASET.PATCH.HEIGHT//2:-CFG.DATASET.PATCH.HEIGHT//2+1,CFG.DATASET.PATCH.HEIGHT//2:-CFG.DATASET.PATCH.HEIGHT//2+1]+1
                mask = np.zeros(target_dataset.gt_raw.shape, dtype="bool")
                #0是忽略的类别
                mask[target_dataset.gt_raw == 0] = True
                prediction[mask] = 0
                color_prediction = convert_to_color(prediction,data_name=CFG.DATASET.NAME)
                display_predictions(
                    color_prediction,
                    gt=convert_to_color(target_dataset.gt_raw,data_name=CFG.DATASET.NAME),
                    caption="Prediction vs. test ground truth",
                )
                
                flops_FE, params_FE = profile(FE, (torch.randn(1, source_dataset.data.shape[0], 11, 11).cuda(),))
                flops_C1, params_C1 = profile(C1, (torch.randn(1, 512).cuda(),))
                flops_C2, params_C2 = profile(C2, (torch.randn(1, 512).cuda(),))
                print('flops: %.2f M, params: %.2f M' % ((flops_FE+flops_C1+flops_C2) / 1e6, (params_FE+params_C1+params_C2) / 1e6))            
                return

        source_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer_fe.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        FE.train()  # set model to training mode
        C1.train()
        C2.train()
        metric.reset()  # reset metric
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        step1_loss_epoch, step2_loss_epoch, step3_loss_epoch = 0., 0., 0.
        for iteration in train_bar:
            x_s, label = next(source_iterator)
            x_t, _ = next(target_iterator)
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)

            optimizer_fe.zero_grad()
            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            with autocast():
                # step1: train FE、C1 and C2
                f_s = FE(x_s)
                p1_s, p2_s = C1(f_s)[-1], C2(f_s)[-1]
                step1_loss = cls_criterion(p1_s, label) + cls_criterion(p2_s, label)
            step1_loss_epoch += step1_loss.item()

            scaler.scale(step1_loss).backward()
            scaler.step(optimizer_fe)
            scaler.step(optimizer_c1)
            scaler.step(optimizer_c2)
            scaler.update()

            # step2: fix FE then train C1 and C2
            FE.eval()
            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            with autocast():
                with torch.no_grad():
                    f_s, f_t = FE(x_s), FE(x_t)
                p1_s, p2_s = C1(f_s)[-1], C2(f_s)[-1]
                p1_t, p2_t = C1(f_t)[-1], C2(f_t)[-1]
                cls_loss = cls_criterion(p1_s, label) + cls_criterion(p2_s, label)
                dis_loss = -1 * dis_criterion(p1_t, p2_t)
                step2_loss = cls_loss + dis_loss
            step2_loss_epoch += step2_loss.item()

            scaler.scale(step2_loss).backward()
            scaler.step(optimizer_c1)
            scaler.step(optimizer_c2)
            scaler.update()

            # step3: fix C1 and C2 then train FE
            FE.train()
            C1.eval()
            C2.eval()
            for i in range(CFG.EPOCHK):
                optimizer_fe.zero_grad()
                with autocast():
                    f_t = FE(x_t)
                    p1_t, p2_t = C1(f_t)[-1], C2(f_t)[-1]
                    step3_loss = dis_criterion(p1_t, p2_t)
                step3_loss_epoch += step3_loss
                #.item()

                scaler.scale(step3_loss).backward()
                scaler.step(optimizer_fe)
                scaler.update()

            pred = ((p1_s + p2_s) / 2).argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

            train_bar.set_postfix({
                'epoch': epoch,
                'loss_step1': f'{step1_loss.item():.3f}',
                'loss_step2': f'{step2_loss.item():.3f}',
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}'
            })

        step1_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        step2_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        step3_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE * CFG.EPOCHK
        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()

        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_step1-epoch', step1_loss_epoch, epoch)
            writer.add_scalar('train/loss_step2-epoch', step2_loss_epoch, epoch)
            writer.add_scalar('train/loss_step3-epoch', step3_loss_epoch, epoch)

            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)
        logging.info(
            'rank{} train epoch={} | loss_step1={:.3f} loss_step2={:.3f} loss_step3={:.3f}'.format(
                dist.get_rank() + 1, epoch, step1_loss_epoch, step2_loss_epoch, step3_loss_epoch))
        logging.info(
            'rank{} train epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                      Ps[c], Rs[c], F1S[c]))

        # validate
        if args.no_validate:
            continue
        FE.eval()  # set model to evaluation mode
        C1.eval()  # set model to evaluation mode
        C2.eval()  # set model to evaluation mode
        # 由于 retain graph = true 此处不能用eval() 否则计算图会被free掉 导致模型失效
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label in val_bar:
                x_t, label = x_t.to(device), label.to(device)
                with autocast():
                    f_t = FE(x_t)
                    p1_t, p2_t = C1(f_t)[-1], C2(f_t)[-1]
                    y_t = (p1_t + p2_t) / 2
                    cls_loss = val_criterion(y_t, label)
                val_loss += cls_loss.item()

                pred = y_t.argmax(axis=1)
                metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{cls_loss.item():.3f}',
                    'mP': f'{metric.mPA():.3f}',
                    'PA': f'{metric.PA():.3f}',
                    'KC': f'{metric.KC():.3f}'
                })
        val_loss /= len(val_dataloader) * CFG.DATALOADER.BATCH_SIZE

        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        wandb.log({'PA': PA, 'mPA': mPA,'KC': KC})

        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/PA-epoch', PA, epoch)
            writer.add_scalar('val/mPA-epoch', mPA, epoch)
            writer.add_scalar('val/KC-epoch', KC, epoch)
        if PA > best_PA:
            best_epoch = epoch

        logging.info('rank{} val epoch={} | loss={:.3f}'.format(dist.get_rank() + 1, epoch, val_loss))
        logging.info(
            'rank{} val epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                    Ps[c], Rs[c], F1S[c]))

        # adjust learning rate if specified
        for s in [scheduler1, scheduler2, scheduler3]:
            if s is not None:
                try:
                    s.step()
                except TypeError:
                    s.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'FE': {
                    'state_dict': FE.state_dict(),
                },
                'C1': {
                    'state_dict': C1.state_dict(),
                },
                'C2': {
                    'state_dict': C2.state_dict(),
                },
                'optimizer_fe': {
                    'state_dict': optimizer_fe.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'optimizer_c1': {
                    'state_dict': optimizer_c1.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'optimizer_c2': {
                    'state_dict': optimizer_c2.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'metric': {
                    'PA': PA,
                    'mPA': mPA,
                    'Ps': Ps,
                    'Rs': Rs,
                    'F1S': F1S,
                    'KC': KC
                },
            }
            torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
            if PA > best_PA:
                best_PA = PA
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))
            writer.add_scalar('best-PA', best_PA, epoch)
            wandb.log({'best-PA':best_PA})


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
