# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
from scipy import io, misc
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import mat73
from typing import Optional, Any, Tuple
from torch.autograd import Function
import math
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,y: torch.Tensor,label,name, source_color='r', target_color='b'):
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    if len(y.shape)>1:
        _, y = torch.max(y, dim=1)
    y = y.cpu().detach().numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
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
        plt.Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[4], label=f'Source - Class {class_label}', markersize=16),
        plt.Line2D([0], [0], marker='o',color='w', markerfacecolor=colors[1], label=f'Target - Class {class_label}', markersize=16),
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
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     if len(y.shape)>1:
#         _, y = torch.max(y, dim=1)
#     y = y.cpu().detach().numpy()
#     if(y.min()==0):
#         y = y+1
#     # features = np.concatenate([source_feature, target_feature], axis=0)
#     features = target_feature
#     _,y= np.split(y,2)
#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
#     # domain labels, 1 represents source while 0 represents target
#     # domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#     domains =np.ones(len(target_feature))

#     colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'purple', 6: 'brown', 7: 'cyan'}   
#     label = [1,2,3,4,5,6,7]
        
#     plt.figure(figsize=(10, 8))  
#     plt.xlim((-90, 90))
#     plt.ylim((-90, 90))
#     for i in range(len(X_tsne)):
#         if label!=None:
#             if y[i].item() in label:
#                 if domains[i]==1:
#                     plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
#                                 color=colors[y[i].item()],
#                                 marker='o',
#                                 alpha=0.7)
#                 else:
#                     plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
#                                 edgecolors=colors[y[i].item()],
#                                 marker='o',c="none",alpha=0.7)
#         else:
#             plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
#                             color=colors[y[i].item()],
#                             marker='o',
#                             alpha=0.7)

#     plt.legend(handles=[ plt.Line2D([0], [0], marker='o',color='w', markerfacecolor=colors[i], label=f'Target - Class {i}', markersize=15) for i in label],loc='upper right', fontsize=16)
#     plt.title(f't-SNE Visualization for All Classes')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.grid()
#     # 显示图形
#     plt.show()
#     Img= wandb.Image(plt, caption="TSNE"+name)
#     wandb.log({"TSNE":Img,})
#     plt.close()

class ConsistencyLoss:
        
    def __call__(self, logits, targets, mask=None):
        
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()

class SelfAdaptiveFairnessLoss:
    
    def __call__(self, mask, logits_ulb_s, p_t, label_hist):
        
        # Take high confidence examples based on Eq 7 of the paper
        logits_ulb_s = logits_ulb_s[mask.bool()]
        probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
        max_idx_s = torch.argmax(probs_ulb_s, dim=-1)
        
        # Calculate the histogram of strong logits acc. to Eq. 9
        # Cast it to the dtype of the strong logits to remove the error of division of float by long
        histogram = torch.bincount(max_idx_s, minlength=logits_ulb_s.shape[1]).to(logits_ulb_s.dtype)
        histogram /= histogram.sum()

        # Eq. 11 of the paper.
        p_t = p_t.reshape(1, -1)
        label_hist = label_hist.reshape(1, -1)
        
        # Divide by the Sum Norm for both the weak and strong augmentations
        scaler_p_t = self.__check__nans__(1 / label_hist).detach()
        modulate_p_t = p_t * scaler_p_t
        modulate_p_t /= modulate_p_t.sum(dim=-1, keepdim=True)
        
        scaler_prob_s = self.__check__nans__(1 / histogram).detach()
        modulate_prob_s = probs_ulb_s.mean(dim=0, keepdim=True) * scaler_prob_s
        modulate_prob_s /= modulate_prob_s.sum(dim=-1, keepdim=True)
        
        # Cross entropy loss between two Sum Norm logits. 
        loss = (modulate_p_t * torch.log(modulate_prob_s + 1e-9)).sum(dim=1).mean()
        
        return loss, histogram.mean()

    @staticmethod
    def __check__nans__(x):
        x[x == float('inf')] = 0.0
        return x

class SelfAdaptiveThresholdLoss:
    
    def __init__(self, sat_ema):
        
        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()
        
    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        
        # Updating the histogram for the SAF loss here so that I dont have to call the torch.no_grad() function again. 
        # You can do it in the SAF loss also, but without accumulating the gradient through the weak augmented logits
        
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())
        return tau_t, p_t, label_hist
   
    def __call__(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):

        tau_t, p_t, label_hist = self.__update__params__(logits_ulb_w, tau_t, p_t, label_hist)
        
        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)

        return loss, mask, tau_t, p_t, label_hist


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)

        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                # nn.BatchNorm1d(in_feature),
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                # nn.Linear(hidden_size, hidden_size),
                # nn.BatchNorm1d(hidden_size),
                # nn.ReLU(),
                final_layer
                )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                final_layer
            )
    
class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None, is_weak: Optional[torch.Tensor]=True,myweight=None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)  
        if self.sigmoid:
            d_label = torch.cat((
        torch.ones((f_s.size(0),)).to(f_s.device),
        torch.zeros((f_t.size(0),)).to(f_t.device),))   
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).float()
            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            if myweight!=None:
                # myweight = myweight / torch.sum(myweight) * myweight.shape[0]
                w_s ,w_t    =myweight.chunk(2, dim=0)
            loss = 0.5 * (F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +\
                        F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t).detach(), reduction=self.reduction))
            return loss
           
        else:
            d_label = torch.cat((
            torch.ones((f_s.size(0),)).to(f_s.device),
            torch.zeros((f_t.size(0),)).to(f_t.device),
        )).long()     
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)
            
class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None
    
class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)
    
class ConditionalDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean', eps = 0.0):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning
        self.eps = eps
        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor,grad_fs: torch.Tensor=None,grad_ft: torch.Tensor=None,myweight=None) -> torch.Tensor:
    
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        d_label = torch.cat((
            torch.ones((g_s.size(0), 1)).to(g_s.device) * (1-self.eps),
            torch.ones((g_t.size(0), 1)).to(g_t.device) * self.eps,
        ))
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        if(myweight!=None):
            # myweight              =myweight.cpu().detach().numpy()
            # source_weight,target_weight=weight.chunk(2, dim=0)
            # source_weight=source_weight / torch.sum(source_weight) * batch_size/2
            # target_weight=target_weight*myweight
            # target_weight=target_weight / torch.sum(target_weight) * batch_size/2
            weight=weight*myweight
            # weight=torch.concatenate((source_weight,target_weight),0)
        # else:
        weight = weight / torch.sum(weight) * batch_size
        self.domain_discriminator_accuracy = binary_accuracy(d, torch.where(d_label>0.5, 1, 0))
        return self.bce(d, d_label, weight.view_as(d))
    
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct    
    
def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss
    
class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss
        
def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        try:
            data = io.loadmat(dataset)
        except :
            data = mat73.loadmat(dataset) 
            # data = next(self.iter)
        return data
    #io.loadmat(dataset)
    
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
        Img= wandb.Image(pred, caption=caption)
        wandb.log({"pred":Img,})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})
        Img= wandb.Image(pred, caption=caption)
        gt= wandb.Image(gt, caption=caption)
        wandb.log({"pred":Img, "gt":gt})

def display_dataset(img, gt, bands, labels, palette, vis, name):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption =name+ "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})
    Img= wandb.Image(rgb, caption= name)
    wandb.log({caption:Img,})

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        # wandb.
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def padding_image(image, patch_size=None, mode="symmetric", constant_values=0):
    """Padding an input image.
    Modified at 2020.11.16. If you find any issues, please email at mengxue_zhang@hhu.edu.cn with details.

    Args:
        image: 2D+ image with a shape of [h, w, ...],
        The array to pad
        patch_size: optional, a list include two integers, default is [1, 1] for pure spectra algorithm,
        The patch size of the algorithm
        mode: optional, str or function, default is "symmetric",
        Including 'constant', 'reflect', 'symmetric', more details see np.pad()
        constant_values: optional, sequence or scalar, default is 0,
        Used in 'constant'.  The values to set the padded values for each axis
    Returns:
        padded_image with a shape of [h + patch_size[0] // 2 * 2, w + patch_size[1] // 2 * 2, ...]

    """
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    pad_width = [[h, h], [w, w]]
    [pad_width.append([0, 0]) for i in image.shape[2:]]
    padded_image = np.pad(image, pad_width, mode=mode, constant_values=constant_values)
    return padded_image


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1 *100
    results["F1 scores"] = F1scores

    Accuracy_scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = cm[i, i] / (np.sum(cm[i, :]))
        except ZeroDivisionError:
            F1 = 0.
        Accuracy_scores[i] = F1 *100
    results["Accuracy scores"] = Accuracy_scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    kappa = kappa * 100
    results["Kappa"] = kappa

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]
        class_specific=[r["Accuracy scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        class_specific_scores_mean = np.mean(class_specific, axis=0)
        class_specific_scores_std = np.std(class_specific, axis=0)
        
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]
        class_specific=results["Accuracy scores"]

    vis.heatmap(cm, opts={'title': "Confusion matrix",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    # wandb.Image(cm)
    # # send to visdom server
    # vis.images([np.transpose(rgb, (2, 0, 1))],
    #             opts={'caption': caption})
    # Img= wandb.Image(rgb, caption= name)    
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"
    
    text += "Class-specific scores :\n"
    if agregated:
        for label, score, std in zip(label_values, class_specific_scores_mean,
                                     class_specific_scores_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, class_specific):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.02f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    # wandb.log(text.replace('\n', '<br/>'))

    print(text)
    
def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set

# def sample_gt(gt, train_size, mode='random'):
#     """Extract a fixed percentage of samples from an array of labels.

#     Args:
#         gt: a 2D array of int labels
#         percentage: [0, 1] float
#     Returns:
#         train_gt, test_gt: 2D arrays of int labels

#     """
#     indices = np.nonzero(gt)
#     X = list(zip(*indices)) # x,y features
#     y = gt[indices].ravel() # classes
#     train_gt = np.zeros_like(gt)
#     test_gt = np.zeros_like(gt)
#     if train_size > 1:
#        train_size = int(train_size)

#     if mode == 'random':
#        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
#        train_indices = [list(t) for t in zip(*train_indices)]
#        test_indices = [list(t) for t in zip(*test_indices)]
#        train_gt[train_indices] = gt[train_indices]
#        test_gt[test_indices] = gt[test_indices]
#     elif mode == 'fixed':
#        print("Sampling {} with train size = {}".format(mode, train_size))
#        train_indices, test_indices = [], []
#        for c in np.unique(gt):
#            if c == 0:
#               continue
#            indices = np.nonzero(gt == c)
#            X = list(zip(*indices)) # x,y features

#            train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
#            train_indices += train
#            test_indices += test
#        train_indices = [list(t) for t in zip(*train_indices)]
#        test_indices = [list(t) for t in zip(*test_indices)]
#        train_gt[train_indices] = gt[train_indices]
#        test_gt[test_indices] = gt[test_indices]

#     elif mode == 'disjoint':
#         train_gt = np.copy(gt)
#         test_gt = np.copy(gt)
#         for c in np.unique(gt):
#             mask = gt == c
#             for x in range(gt.shape[0]):
#                 first_half_count = np.count_nonzero(mask[:x, :])
#                 second_half_count = np.count_nonzero(mask[x:, :])
#                 try:
#                     ratio = first_half_count / (first_half_count + second_half_count)
#                     if ratio > 0.9 * train_size:
#                         break
#                 except ZeroDivisionError:
#                     continue
#             mask[:x, :] = 0
#             train_gt[mask] = 0

#         test_gt[train_gt > 0] = 0
#     else:
#         raise ValueError("{} sampling is not implemented yet.".format(mode))
#     return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
