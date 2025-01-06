# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import copy
# utils
from datetime import datetime 
import math
import os
import wandb
import datetime
import numpy as np
import joblib
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake,GraphConvolution,SelfAdaptiveThresholdLoss,SelfAdaptiveFairnessLoss
from torch.optim.lr_scheduler import StepLR
from linformer import Linformer
#from vit_pytorch.vit import ViT
from vit_pytorch.efficient import ViT
from vit_pytorch.local_vit import LocalViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.rvt import RvT
from vit_pytorch.pit import PiT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.levit import LeViT
from vit_pytorch.cvt import CvT
from vit_pytorch.tnt import TNT
from vit_pytorch.vip import VisionPermutator, WeightedPermuteMLP
from vit_pytorch.hit import HiT, ConvPermuteMLP
from vit_pytorch.cait import CaiT
from vit_pytorch.ccvt import CCvT
from RCNN import RCNN
from conv2d import TESTEtAl
from conv3d import C3DEtAl
from yang import Yangnew
from Involution3 import I2DEtAl, I3DEtAl
from vit_pytorch.vipp import ViP
from vit_pytorch.dwt import DWT, ConvPermute
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv

def euclidean_distance(a, b):
    # 计算欧式距离
    dist = torch.sqrt(torch.sum(torch.square(a[:, None] - b), axis=2))
    return dist
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # if (iter_num / max_iter)<0.5:
    #     ret=2*(iter_num / max_iter)**2
    # else:
    #     ret=1-2*(1-iter_num / max_iter)**2
    # return ret  
    # return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
    return np.float64(iter_num / max_iter)  

#iter_num/max_iter
#np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
    
def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "nn":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes, kwargs.setdefault("dropout", False))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == 'yang':
        kwargs.setdefault('patch_size', 15)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.1)
        model = Yangnew(n_bands, n_classes, patch_size=15)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == "vit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 15)
        ThreeDModel = HamidaEtAl(n_bands, n_classes, patch_size=7)

        efficient_transformer = Linformer(
            dim=128,
            seq_len=25 + 1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        model = ViT(
            dim=128,
            image_size=15,
            patch_size=3,
            num_classes=n_classes,
            transformer=efficient_transformer,
            channels=n_bands,
            ThreeDModel=ThreeDModel
        )
       # model = ViT(dim=1024, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0003)
        ### lr for efficient transformer
        #lr = kwargs.setdefault("learning_rate", 0.005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "lvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = LocalViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "t2t":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = T2TViT(dim=512, image_size=15, depth=4, heads=8, mlp_dim=512, t2t_layers=((7, 2), (3, 2), (3, 2)), dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "rvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = RvT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0003)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "pit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = PiT(dim=256, image_size=15, patch_size=3, depth=(3, 3, 3, 3), heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "dit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = DeepViT(dim=1024, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "crosvit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CrossViT(image_size=15, num_classes=n_classes, channels=n_bands, sm_dim=192, lg_dim=384, )
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "levit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = LeViT(dim=(256, 384, 512), image_size=224, stages=3,  depth=4, heads=(4, 6, 8), mlp_mult=2, dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "cvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "ccvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CCvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "cait":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CaiT(image_size=15, patch_size=3, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tnt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TNT(image_size=15, patch_dim=512, pixel_dim=24, patch_size=3, pixel_size=3, depth=6, heads=16, attn_dropout=0.1, ff_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "vip":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        transitions = [False, True, False, False]
        segment_dim = [8, 8, 4, 4]
        mlp_ratios = [3, 3, 3, 3]
        embed_dims = [256, 256, 512, 512]
        model = VisionPermutator(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "vipp":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = ViP(pretrained=False, in_chans=n_bands,inplanes=64, num_classes=n_classes, num_chs=(96, 192, 384, 768), patch_sizes=[1, 1, 1, 1], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 64], num_layers=[1, 1, 3, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0.1,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "dwt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        transitions = [False, True, False, False]
        segment_dim = [8, 8, 4, 4]
        mlp_ratios = [3, 3, 3, 3]
        embed_dims = [256, 256, 512, 512]
        model = DWT(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermute,)
        lr = kwargs.setdefault("learning_rate", 0.0001)## for KSC 0.000003
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 64)
    elif name == "hit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        transitions = [False, True, False, False]
        segment_dim = [8, 8, 4, 4]
        mlp_ratios = [3, 3, 3, 3]
        embed_dims = [480, 480, 512, 512]## for IN 368, for GRSS 256, for PU 168, for KSC 320 for XA 480
        model = HiT(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermuteMLP,)
        lr = kwargs.setdefault("learning_rate", 0.0001)## for KSC 0.000003
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 64)
    elif name == "rcnn":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = RCNN(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "conv2d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TESTEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "conv3d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = C3DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "i2d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = I2DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "i3d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = I3DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "hamida":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", kwargs["lr"])
        
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "lee":
        kwargs.setdefault("epoch", 200)
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "chen":
        patch_size = kwargs.setdefault("patch_size", 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 400)
        kwargs.setdefault("batch_size", 100)
    elif name == "li":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        epoch = kwargs.setdefault("epoch", 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "he":
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault("patch_size", 7)
        kwargs.setdefault("batch_size", 40)
        lr = kwargs.setdefault("learning_rate", 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "luo":
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault("patch_size", 3)
        kwargs.setdefault("batch_size", 100)
        lr = kwargs.setdefault("learning_rate", 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "sharma":
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault("batch_size", 60)
        epoch = kwargs.setdefault("epoch", 30)
        lr = kwargs.setdefault("lr", 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault("patch_size", 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1
            ),
        )
    elif name == "liu":
        kwargs["supervision"] = "semi"
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault("epoch", 40)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()
            ),
        )
    elif name == "boulch":
        kwargs["supervision"] = "semi"
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("epoch", 100)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(rec, data.squeeze()),
        )
    elif name == "mou":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        kwargs.setdefault("epoch", 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault("lr", 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 100)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        # x = self.fc4(x)
        return x,self.fc4(x)


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.conv2 = nn.Conv3d(
            20, 20, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # self.conv3 = nn.Conv3d(
        #     35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        # )
        self.conv4 = nn.Conv3d(
            20, 8, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        self.dropout = nn.Dropout(p=0.1)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        # self.gcn = GraphConvolution(self.features_size, self.features_size)
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)
        # self.poolformap=nn.MaxPool3d((3, 7, 7), stride=(2, 4, 4))
        # self.upsample =nn.Conv3d( 8, 1, (7, 1, 1), dilation=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2))
        # self.convadv = nn.Conv2d(7, 1, (3, 3), stride=(1, 1))
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            # x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, adj=None):
        if(adj==None):
            adj=torch.eye(len(x)).cuda()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(1)
        # x = self.gcn(x, adj)
        x = self.dropout(x)
        # x = self.fc(x)
        return x, self.fc(x)
    
    def feature_extract(self, x, adj=None):
        if(adj==None):
            adj=torch.eye(len(x)).cuda()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = torch.squeeze(x)
        x = self.convadv(x)
        x = torch.squeeze(x)
        
        # x = self.poolformap(x)
        # x = x.flatten(1)
        # # x = self.gcn(x, adj)
        # x = self.dropout(x)
        # f = torch.mm(x, self.map.to(x.device))
        # x = self.fc(x)
        return x

class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        # x = self.conv8(x)
        return x,self.conv8(x)


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(
            self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        )
        x = F.relu(
            self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        )
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=15):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while n > 1:
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        return x,self.fc(x)


def train(
    net,
    Align_dict,
    optimizer,
    criterion,
    data_loader,
    data_loader_t,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    hyperparams=None,
    val_loader=None,
    supervision="full",
    m_class=None
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    # cls_accs = AverageMeter('Cls Acc', ':3.1f')
    # target_cls_accs = AverageMeter('target_cls_accs Acc', ':3.1f')
    # progress = ProgressMeter(
    #     [cls_accs,target_cls_accs],prefix="Epoch: [{}]".format(epoch))
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(1000000)
    dann_losses = np.zeros(1000000)
    cdan_losses = np.zeros(1000000)

    mean_dann_losses = np.zeros(1000000)
    mean_cdan_losses = np.zeros(1000000)

    iter_ = 1
    gamma = 0.9
    scheduler = StepLR(optimizer, step_size=4, gamma=gamma)
    loss_win, val_win,loss_dann,loss_cdan = None, None, None,None
    val_accuracies = []
    myval_accuracies = []
    total=hyperparams["training_times"]
    #len(data_loader)
    # total=hyperparams["training_times"]
    iter_data=ForeverDataIterator(data_loader)
    iter_data_t=ForeverDataIterator(data_loader_t)
    # if hyperparams["na"]==True:
    mem_fea = torch.rand(len(iter_data_t.data_loader.dataset), net.features_size).cuda()
    mem_ord = torch.rand(len(iter_data_t.data_loader.dataset), 2).cuda()
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls = torch.ones(len(iter_data_t.data_loader.dataset), hyperparams["n_classes"]).cuda() / hyperparams["n_classes"]
    sat_criterion = SelfAdaptiveThresholdLoss(0.999)
    saf_criterion = SelfAdaptiveFairnessLoss()
    p_t = torch.ones(m_class).cuda() / m_class
    label_hist = torch.ones(m_class).cuda() / m_class
    tau_t = p_t.mean()
    source_p_t = torch.ones(m_class).cuda() / m_class
    source_label_hist = torch.ones(m_class).cuda() / m_class
    source_tau_t = p_t.mean()
    
    for e in tqdm(range(1, epoch + 1), desc="Training epoch"):
        # Set the network to training mode
        net.train()
        Align_dict['dann'].to(device)
        Align_dict['dann'].train()
        Align_dict['cdan'].to(device)
        Align_dict['cdan'].train()
        avg_loss = 0.0
        
        # Run the training loop for one epoch
        # for batch_idx, (data, target) in tqdm(
        #     enumerate(data_loader), total=len(data_loader)
        # ):
        # eff = e / epoch + 1
        acc_raw=0
        acc_nc =0
        for batch_idx in tqdm(range(0, total), desc="Training the network"):            
            data_raw, data, target,ord_s,weight_s,_=next(iter_data)
            data_t_raw,data_t, target_t,ord_t,weight_t,idx=next(iter_data_t)
            eff = calc_coeff(max(max(e,0)*total+batch_idx-len(val_loader),0), max_iter=total*epoch)

            # Load the data into the GPU if required
            x_s, labels_s,ord_s,weight_s = data.to(device), target.to(device),ord_s.to(device),weight_s.to(device)
            x_t_raw, x_t, labels_t,ord_t,weight_t = data_t_raw.to(device), data_t.to(device), target_t.to(device),ord_t.to(device),weight_t.to(device)
            weight_s = (100 * (weight_s+3) / (torch.sum((weight_s+3))+0.001))
            # labels_s  =  labels_s-1
            # labels_t  =  labels_t-1
            data = torch.cat((x_s, x_t,x_t_raw), dim=0)

            optimizer.zero_grad()
            if supervision == "full":
                # adj_s    = torch.exp(-torch.cdist(ord_s.float(), ord_s.float(), p=2))
                # adj_t    = torch.exp(-torch.cdist(ord_t.float(), ord_t.float(), p=2))
                f,output = net(data)
                # f        = net.feature_extract(data_adv)
                features_source, features_target,_ = f.chunk(3, dim=0)
                output_s, output_t,_ = output.chunk(3, dim=0)
                #\
                # +hyperparams['dann']*Align_dict['dann'](features_source, features_target)\
                # +hyperparams['cdan']*Align_dict['cdan'](output_s, features_source, output_t, features_target)
                # if hyperparams["na"]==True:
                dis = -torch.mm(features_target.detach(), mem_fea.t())
                dis_ord = euclidean_distance(ord_t.detach(),mem_ord.detach())
                for di in range(dis.size(0)):
                    dis[di, idx[di]] = torch.max(dis)  #去除自己本身的
                    dis_ord[di, idx[di]] = torch.max(dis_ord)  #去除自己本身的
                _, p1 = torch.sort(dis*1+dis_ord*hyperparams["ratio_ord"], dim=1)

                w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(hyperparams["group"]):
                        w[wi][p1[wi, wj]] = 1/ hyperparams["group"]

                # weight_, pred = torch.max(w.mm(mem_cls), 1)

                # # if hyperparams["na_now"] == True:
                # #     classifier_loss = nn.CrossEntropyLoss()(output_t, pred) 
                # # else:
                # loss_ = nn.CrossEntropyLoss(reduction='none')(output_t, pred)
                # classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())
                # loss += hyperparams["na"]*eff * classifier_loss
                
                
                # img_ulb_w, img_ulb_s = data_t_raw, data_t
                source_logits_ulb_w,source_logits_ulb_s = output_s,output_s
                
                logits_ulb_w, logits_ulb_s = w.mm(mem_cls),output_t
                
                # loss_sat, mask, tau_t, p_t, label_hist = sat_criterion(
                #     logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist
                # )
                _, _, tau_t, p_t, label_hist = sat_criterion(
                    logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist
                )                
                _, _, source_tau_t, source_p_t, source_label_hist = sat_criterion(
                    source_logits_ulb_w, source_logits_ulb_s, source_tau_t, source_p_t, source_label_hist
                )   
                newlabel_hist=(label_hist + 1)/(source_label_hist + 1)
                
                # loss_saf, hist_p_ulb_s = saf_criterion(mask, logits_ulb_s, p_t, label_hist)   
                # loss +=hyperparams["sat"]*loss_sat*eff+hyperparams["saf"]*loss_saf

                weight_setc, pred = torch.max(w.mm(mem_cls), 1)
                weight_setc = (100 * weight_setc / (torch.sum(weight_setc)+0.001))
                weight_source     = newlabel_hist[labels_s]
                weight_source = (100 * weight_source / (torch.sum(weight_source)+0.001))
                # weight_=weight_/label_hist[pred]
                # if hyperparams["na_now"] == True:
                #     classifier_loss = nn.CrossEntropyLoss()(output_t, pred) 
                # else:
                SETC_loss = nn.CrossEntropyLoss(reduction='none',weight=hyperparams["weights"])(output_t, pred)
                SETC_loss = (weight_setc * SETC_loss).mean()
                # SETC_loss = torch.sum(weight_setc * SETC_loss) / (torch.sum(weight_setc).item())
                classifier_loss = nn.CrossEntropyLoss(reduction='none',weight=hyperparams["weights"],label_smoothing=0.1)(output_s, labels_s)
                if (hyperparams["class_match"]==1):
                    classifier_loss = torch.sum(weight_source * classifier_loss) / (torch.sum(weight_source).item())
                else:
                    classifier_loss = classifier_loss.mean()
                # classifier_loss = torch.sum(weight_source * classifier_loss) / (torch.sum(weight_source).item())
                # classifier_loss = (weight_source * classifier_loss).mean()
                # classifier_loss = (classifier_loss).mean()

                # (criterion(output_s, labels_s)*weight_s).mean()
                loss =classifier_loss.mean() + hyperparams["na"]*eff * SETC_loss.mean() + hyperparams['mcc']*Align_dict['mcc'](output_t)
                
                # net.eval() 
                with torch.no_grad():
                    features_target, outputs_target = net(x_t)
                    features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                    softmax_out = nn.Softmax(dim=1)(outputs_target)
                    # if args.pl == 'atdoc_na_nos':
                    outputs_target = softmax_out
                    # else:
                    # outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
                momentum=0.99
                mem_fea[idx] = (1.0 - momentum) * mem_fea[idx] + momentum * features_target.clone()
                mem_cls[idx] = (1.0 - momentum) * mem_cls[idx] + momentum * outputs_target.clone()
                mem_ord[idx] = ord_t.float()

            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if display_iter and iter_ % display_iter == 0:
                wandb.log({'iteration':(e-1)*total+batch_idx, 'loss': loss, 'DANN_acc': Align_dict['dann'].domain_discriminator_accuracy,
            'CDAN_acc': Align_dict['cdan'].domain_discriminator_accuracy,})
            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= total
        if val_loader is not None:
            # val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_acc, myval_acc= myval(net, val_loader, mem_fea, mem_ord,mem_cls,hyperparams, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            myval_accuracies.append(myval_acc)
            wandb.log({'val_accs':val_acc,'myval_accs':myval_acc,"eff":eff,"best_val":np.array(val_accuracies).max(),"my_best_val":np.array(myval_accuracies).max()  })
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()
        print(scheduler.get_last_lr())
        if val_acc>=np.array(val_accuracies).max():
            model_dir = "./checkpoints/" + camel_to_snake(str(net.__class__.__name__)) + "/" + data_loader.dataset.name + "/"
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            torch.save(net.state_dict(), model_dir + "best" + ".pth")    
        if myval_acc>=np.array(myval_accuracies).max():
            mem_fea_best=copy.deepcopy(mem_fea)
            mem_ord_best=copy.deepcopy(mem_ord)
            mem_cls_best=copy.deepcopy(mem_cls)
            net_best    =copy.deepcopy(net)

    best_model = torch.load(model_dir + "best" + ".pth")
    net.load_state_dict(best_model)
    pre = pred_output(net_best, val_loader,mem_fea_best, mem_ord_best,mem_cls_best,hyperparams, device=device, supervision=supervision)
    return np.array(val_accuracies).max(),np.array(myval_accuracies).max(),pre

def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
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
                #b, d, h, w = data.shape
                #data_z = np.zeros([b, 256, h, w])
                #data_z[:, 0:d, :, :] = data
                #data_z = np.asarray(np.copy(data_z), dtype="float32")
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            f,output = net(data)
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

def Mytest(img,preds):
    """
    Test a model on a specific image
    """
    probs = np.zeros(img.shape[:2])
    for ord, out in preds:
        probs[ord[0] , ord[1]]= out
    return probs

def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    net.eval()
    ignored_labels = data_loader.dataset.ignored_labels
    return_f=[]
    return_y=[]
    return_label= []
    for batch_idx, (data_raw, data, target, ord,weight,_) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target,ord = data.to(device), target.to(device),ord.to(device)
            # target= target-1
            if supervision == "full":
                adj    = torch.exp(-torch.cdist(ord.float(), ord.float(), p=2))
                fea,output = net(data)
            elif supervision == "semi":
                fea,outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # output    = output+1
            for pred, out, f in zip(output.view(-1), target.view(-1),fea):
                if out.item() in ignored_labels:
                    # accuracy += out.item() == pred.item()
                    # total += 1
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
                    return_f.append(f.detach())
                    return_y.append(pred.item())
                    return_label.append(out.item())
                    
    return accuracy / total, torch.stack(return_f),torch.tensor(return_y),torch.tensor(return_label)

def myval(net, data_loader, mem_fea, mem_ord,mem_cls,hyperparams, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    net.eval()
    accuracy, total,accuracy_,total_ = 0.0, 0.0,0.0,0.0
    ignored_labels = data_loader.dataset.ignored_labels
    pbar =tqdm(enumerate(data_loader),desc="test epoch")
    for batch_idx, (data_raw,data, target,ord,weight,idx) in pbar:
        pbar.set_description('Processing '+str(len(data_loader)))
        with torch.no_grad():
            # starttime = datetime.datetime.now()
            # Load the data into the GPU if required
            data, target,ord = data.to(device), target.to(device),ord.to(device)
            # target= target-1
            if supervision == "full":
                # adj    = torch.exp(-torch.cdist(ord.float(), ord.float(), p=2))
                f,output = net(data)
                # print ((datetime.datetime.now() - starttime).microseconds)
                dis = -torch.mm(f.detach(), mem_fea.t())
                dis_ord = euclidean_distance(ord.detach(),mem_ord.detach())
                for di in range(dis.size(0)):
                    dis[di, idx[di]] = torch.max(dis)  #去除自己本身
                    dis_ord[di, idx[di]] = torch.max(dis_ord)  #去除自己本身
                _, p1 = torch.sort(dis*1+dis_ord*hyperparams["ratio_ord"], dim=1)
                # print ((datetime.datetime.now() - starttime).microseconds)
                w = torch.zeros(f.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(hyperparams["group"]):
                        w[wi][p1[wi, wj]] = 1/ hyperparams["group"]

                _, preds = torch.max(w.mm(mem_cls), 1)
                # preds=output
            elif supervision == "semi":
                f,outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # print ((datetime.datetime.now() - starttime).microseconds)
            # output    = output+1
            for pred, pred_, out in zip(output.view(-1),preds.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    # accuracy += out.item() == pred.item()
                    # total += 1
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    accuracy_ += out.item() == pred_.item()
                    total_ += 1
                    total += 1
            # for pred, out in zip(preds.view(-1), target.view(-1)):
            #     if out.item() in ignored_labels:
            #         # accuracy += out.item() == pred.item()
            #         # total += 1
            #         continue
            #     else:
            #         accuracy_ += out.item() == pred.item()
            #         total_ += 1
    return accuracy / total,accuracy_/total_

def pred_output(net, data_loader, mem_fea, mem_ord,mem_cls,hyperparams, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total,accuracy_,total_ = 0.0, 0.0,0.0,0.0
    ignored_labels = data_loader.dataset.ignored_labels
    return_pred_output,return_ords_output=[], []
    for batch_idx, (data_raw,data, target,ord,weight,idx) in tqdm(enumerate(data_loader),desc="test epoch"):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target,ord = data.to(device), target.to(device),ord.to(device)
            # target= target-1
            if supervision == "full":
                adj    = torch.exp(-torch.cdist(ord.float(), ord.float(), p=2))
                f,output = net(data)
                
                dis = -torch.mm(f.detach(), mem_fea.t())
                dis_ord = euclidean_distance(ord.detach(),mem_ord.detach())
                for di in range(dis.size(0)):
                    dis[di, idx[di]] = torch.max(dis)  #去除自己本身的
                    dis_ord[di, idx[di]] = torch.max(dis_ord)  #去除自己本身的
                _, p1 = torch.sort(dis*1+dis_ord*hyperparams["ratio_ord"], dim=1)

                w = torch.zeros(f.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(hyperparams["group"]):
                        w[wi][p1[wi, wj]] = 1/ hyperparams["group"]

                weight_, preds = torch.max(w.mm(mem_cls), 1)
                return_pred_output.append(preds)
                return_ords_output.append(ord)
            
            # output    = output+1
            for pred,pred_, out in zip(output.view(-1),preds.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    # accuracy += out.item() == pred.item()
                    # total += 1
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    accuracy_ += out.item() == pred_.item()
                    total_ += 1
                    total += 1
            # for pred, out in zip(preds.view(-1), target.view(-1)):
            #     if out.item() in ignored_labels:
            #         # accuracy += out.item() == pred.item()
            #         # total += 1
            #         continue
            #     else:
            #         accuracy_ += out.item() == pred.item()
            #         total_ += 1
    return zip(torch.cat(return_ords_output),torch.cat(return_pred_output))