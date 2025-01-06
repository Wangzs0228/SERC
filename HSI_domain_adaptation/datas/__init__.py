import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader
from tllib.utils.data import ForeverDataIterator
from util.utils import grouper, sliding_window, count_sliding_window, camel_to_snake,convert_to_color_,display_predictions
from configs import CFG
from datas.base import DynamicDataset
from datas.datasets import HoustonDataset, HyRankDataset, ShangHangDataset, PaviaDataset, IndianaDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col
import torch
import numpy as np
import wandb
from tqdm import tqdm

def test(net, img, patch_size,batch_size,n_classes,step=1):
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
            try:
                f,output = net(data)
            except :
                _, output, _ = net(data, alpha=0) 

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



def convert_to_color(x, data_name):
    palette={0: (0, 0, 143),1:(0,31,255),2:(0,175,255),3:(63,255,191),4:(219,255,41),5:(255,159,0),6:(255,15,0),7:(127,0,0)}
    if data_name=="ShangHang":
        palette={0: (0, 0, 143),1:(0,0,143),2:(143,255,111),3:(127,0,0)}   
    return convert_to_color_(x, palette=palette)



def extract(net, data_loader, device):
    net.eval()
    return_f = []
    return_y = []
    return_label = []

    for i, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            fea, outs = net(data)
            return_f.append(fea.detach())
            return_y.append(outs.detach())
            return_label.append(target.detach())

    r_f = torch.cat(return_f)
    r_y = torch.cat(return_y)
    r_l = torch.cat(return_label)

    if r_l.min()==0:
        r_l = r_l +1 
    return r_f, r_y ,r_l
            
def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,y: torch.Tensor,label,name, source_color='r', target_color='b'):
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    if len(y.shape)>1:
        _, y = torch.max(y, dim=1)
    y = y.cpu().detach().numpy()
    if y.min()==0:
        y=y+1
    features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'purple', 6: 'brown', 7: 'cyan'}   
    unique_classes = np.unique(y)
    
    for class_label in unique_classes:
        plt.figure(figsize=(10, 8))
        indices = np.where(y == class_label)
        X_class = X_tsne[indices]
        domain_class = domains[indices]
        # 绘制每个点
        for i in range(len(X_class)):
            if domain_class[i]==1:
                plt.scatter(X_class[i, 0], X_class[i, 1],
                            color=colors[4],
                            marker='o',
                                alpha=0.7)
            else:
                # plt.scatter(X_class[i, 0], X_class[i, 1],
                #             edgecolors=colors[class_label],marker='o',c="none",
                #             alpha=0.7)
                plt.scatter(X_class[i, 0], X_class[i, 1],
                            color=colors[1],
                            marker='o',
                                alpha=0.7)
        plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[4], label=f'Source - Class {int(class_label)}', markersize=16),
        plt.Line2D([0], [0], marker='o',color='w', markerfacecolor=colors[1], label=f'Target - Class {int(class_label)}', markersize=16),
    ] ,loc='upper right', fontsize=15)
        # 添加标题和标签
        plt.title(f't-SNE Visualization for Class {class_label}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        # 显示图形
        plt.show()
        Img= wandb.Image(plt, caption=name + f"{class_label}_class.jpg")
        wandb.log({f"class_{class_label}.jpg":Img,})
        plt.close()
        
    plt.figure(figsize=(10, 8))  
    for i in range(len(X_tsne)):
        if label!=None:
            if y[i].item() in label:
                if domains[i]==1:
                    plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                                color=colors[y[i].item()],
                                marker='o',
                                alpha=0.7)
                else:
                    plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                                edgecolors=colors[y[i].item()],
                                marker='o',c="none",alpha=0.7)
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                            color=colors[y[i].item()],
                            marker='o',
                            alpha=0.7)

    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[i], label=f'Source - Class {i}', markersize=15) for i in unique_classes]+
               [ plt.Line2D([0], [0], marker='o',color='w', markeredgecolor=colors[i], label=f'Target - Class {i}', markersize=15) for i in unique_classes],loc='upper right', fontsize=16)
    plt.title(f't-SNE Visualization for All Classes')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    # 显示图形
    plt.show()
    Img= wandb.Image(plt, caption="TSNE")
    wandb.log({"TSNE":Img,})
    plt.close()
    
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,y: torch.Tensor,label,name, source_color='r', target_color='b'):
#     """
#     Visualize features from different domains using t-SNE.

#     Args:
#         source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
#         target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
#         filename (str): the file name to save t-SNE
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'

#     """
#     _, y = torch.max(y, dim=1)
#     y  = y.cpu().numpy()
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)

#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # domain labels, 1 represents source while 0 represents target
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#     colors = {0: "b", 1: "r"}
#     unique_classes = np.unique(y)

#     for class_label in unique_classes:
#         plt.figure(figsize=(10, 8))
#         # 筛选当前类别的数据点
#         indices = np.where(y == class_label)
#         X_class = X_tsne[indices]
#         domain_class = domains[indices]
        
#         # 绘制每个点
#         for i in range(len(X_class)):
#             plt.scatter(X_class[i, 0], X_class[i, 1],
#                         color=colors[domain_class[i]],
#                         alpha=0.7)
#         plt.legend(handles=[
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Source - Class {class_label}', 
#                     markerfacecolor=colors[1], markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Target - Class {class_label}', 
#                     markerfacecolor=colors[0], markersize=10),
#     ] ,loc='upper right', fontsize=10)
#         # 添加标题和标签
#         plt.title(f't-SNE Visualization for Class {class_label}')
#         plt.xlabel('t-SNE Component 1')
#         plt.ylabel('t-SNE Component 2')
#         plt.grid()
#         # 显示图形
#         plt.show()
#         plt.savefig(f"{class_label}_class.jpg")
#         Img= wandb.Image(plt, caption=name + f"{class_label}_class.jpg")
#         wandb.log({f"class_{class_label}.jpg":Img,})
#         plt.close()
        
def build_transform():
    if CFG.DATASET.NAME in ['Houston', 'HyRANK', 'ShangHang', 'Pavia', 'Indiana']:
        transform = transforms.Compose([
            transforms.LabelRenumber(),
            transforms.ZScoreNormalize(),
            transforms.ToTensor(),
            transforms.FFTCut(CFG.DATASET.FFT_MODE, CFG.DATASET.LOW_PERCENT, CFG.DATASET.HIGH_PERCENT),
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    assert split in ['train', 'val', 'test', 'dynamic']
    if split == 'dynamic':
        return DynamicDataset()
    if CFG.DATASET.NAME == 'Houston':
        dataset = HoustonDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                 CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                 CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'HyRANK':
        dataset = HyRankDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'ShangHang':
        dataset = ShangHangDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                   CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                   CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'Pavia':
        dataset = PaviaDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                               CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                               CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'Indiana':
        dataset = IndianaDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                 CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                 CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())

    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


def build_dataloader(dataset, sampler=None, drop_last=True):
    return DataLoader(dataset,
                      batch_size=CFG.DATALOADER.BATCH_SIZE // dist.get_world_size(),
                      num_workers=CFG.DATALOADER.NUM_WORKERS,
                      pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                      sampler=sampler,
                      drop_last=drop_last
                      )


def build_iterator(dataloader: DataLoader):
    return ForeverDataIterator(dataloader)
