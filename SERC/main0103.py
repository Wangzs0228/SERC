# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division
import torch.optim as optim

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from sklearn import neighbors
from skimage import io
import torch.nn as nn
import wandb

# Visualization
import seaborn as sns
import visdom

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
    MMD_loss,
    CrossEntropyLabelSmooth,
    MinimumClassConfusionLoss,
    ConditionalDomainAdversarialLoss,
    DomainAdversarialLoss,
    DomainDiscriminator,
    visualize
)
from datasets103 import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models0103 import get_model, train, test, save_model,Mytest
from models0313 import val
import torch.backends.cudnn as cudnn
import random
import argparse
    
dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on" " various hyperspectral datasets")
parser.add_argument("--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use.")
parser.add_argument( "--target_dataset", type=str, default=None, choices=dataset_names, help="Target Dataset to use.")
parser.add_argument("--model",type=str,default=None,help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)
parser.add_argument( "--folder",type=str,help="Folder where to store the ""datasets (defaults to the current working directory).",default="./Datasets/",)
parser.add_argument("--cuda", type=int,default=-1,help="Specify CUDA device (defaults to -1, which learns on CPU)",)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument("--restore",type=str,default=None,help="Weights to use for initialization, e.g. a checkpoint",)
parser.add_argument("--seed", type=int, default=0, help="Number of runs (default: 1)")

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument("--training_sample",type=float,default=10,help="Percentage of samples to use for training (default: 10%%)",)
group_dataset.add_argument("--sampling_mode",type=str,help="Sampling mode" " (random sampling or disjoint, default: random)",default="random",)
group_dataset.add_argument("--train_set",type=str,default=None,help="Path to the train ground truth (optional, this ""supersedes the --sampling_mode option)",)
group_dataset.add_argument("--test_set",type=str,default=None,help="Path to the test set (optional, by default ""the test_set is the entire ground truth minus the training)",)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument("--epoch",type=int,help="Training epochs (optional, if" " absent will be set by the model)",)
group_train.add_argument("--patch_size",type=int,help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)",)
group_train.add_argument("--lr", type=float, help="Learning rate, set by the model if not specified.")
group_train.add_argument("--class_balancing",action="store_true",help="Inverse median frequency class balancing (default = False)",)
group_train.add_argument("--batch_size",type=int,help="Batch size (optional, if absent will be set by the model",)
group_train.add_argument( "--test_stride",type=int,default=1,help="Sliding window step stride during inference (default = 1)",)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument("--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)")
group_da.add_argument("--radiation_augmentation",action="store_true", help="Random radiation noise (illumination)",)
group_da.add_argument("--mixture_augmentation", action="store_true", help="Random mixes between spectra")
parser.add_argument("--with_exploration", action="store_true", help="See data exploration visualization")
parser.add_argument("--download",type=str,default=None,nargs="+",choices=dataset_names,help="Download the specified datasets and quits.",)
group_train.add_argument("--mmd",type=float,default=1,)
group_train.add_argument("--mcc",type=float,default=1,)
group_train.add_argument("--dann",type=float,default=1,)
group_train.add_argument("--cdan",type=float,default=1,)
group_train.add_argument("--na",type=float,default=0.05,)
group_train.add_argument("--sat",type=float,default=0.05,)
group_train.add_argument("--saf",type=float,default=0.01,)
group_train.add_argument("--group",type=int,default=8,)
group_train.add_argument("--ratio_ord",type=float,default=0.3,)
group_train.add_argument("--training_times",type=int,default=200,)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
TARGET_DATASET = args.target_dataset

# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
wandb.init(project="SERC107", name=DATASET + " " + MODEL)
wandb.config.update(args,allow_val_change=True)

if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
img_t, gt_t, LABEL_VALUES_T, IGNORED_LABELS_T, RGB_BANDS_T, palette_t = get_dataset(TARGET_DATASET, FOLDER)
# IGNORED_LABELS=[]
# IGNORED_LABELS_T=[]
# gt=gt-1
# gt_t=gt_t-1
# Number of classes
sample_num_src = len(np.nonzero(gt)[0])
sample_num_tar = len(np.nonzero(gt_t)[0])
re_ratio=5
training_sample_ratio=0.05
tmp = training_sample_ratio*re_ratio*sample_num_src/sample_num_tar
training_sample_tar_ratio = tmp if tmp < 1 else 1
r = int(PATCH_SIZE/2)+1
img=np.pad(img,((r,r),(r,r),(0,0)),'symmetric')
img_t=np.pad(img_t,((r,r),(r,r),(0,0)),'symmetric')
gt=np.pad(gt,((r,r),(r,r)),'constant',constant_values=(0,0))
gt_t=np.pad(gt_t,((r,r),(r,r)),'constant',constant_values=(0,0))  

# train_gt_src, _, training_set, _ = sample_gt(gt, training_sample_ratio, mode='random')
# test_gt_tar, _, tesing_set, _ = sample_gt(gt_t, 1, mode='random')
# train_gt_tar, _, _, _ = sample_gt(gt_t, training_sample_tar_ratio, mode='random')
# img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img, img_t, train_gt_src, train_gt_tar
# if tmp < 1:
#     for i in range(re_ratio-1):
#         img_src_con = np.concatenate((img_src_con,img))
#         train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
#         img_tar_con = np.concatenate((img_tar_con,img_t))
#         train_gt_tar_con = np.concatenate((train_gt_tar_con,gt_t))

# train_gt, _,_,_ = sample_gt(gt, 1, mode="random")
# val_gt, _,_,_  = sample_gt(gt_t, 1, mode='random')
# img         = img_src_con
train_gt, train_t_gt, val_gt    = gt,  gt_t,  gt_t
N_CLASSES = len(LABEL_VALUES)+1
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
LABEL_VALUES.insert(0,"0")
# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
# hyperparams.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz, "gt")
display_dataset(img_t, gt_t, RGB_BANDS_T, LABEL_VALUES, palette, viz, "gt_t")

color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

results,results_na = [],[]
# run the experiment several times
for run in range(N_RUNS):
    # if TRAIN_GT is not None and TEST_GT is not None:
    #     train_gt = open_file(TRAIN_GT)
    #     test_gt = open_file(TEST_GT)
    # elif TRAIN_GT is not None:
    #     train_gt = open_file(TRAIN_GT)
    #     test_gt = np.copy(gt)
    #     w, h = test_gt.shape
    #     test_gt[(train_gt > 0)[:w, :h]] = 0
    # elif TEST_GT is not None:
    #     test_gt = open_file(TEST_GT)
    # else:
    #     # Sample random training spectra
    #     train_gt,_, test_gt,_ = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)

    # print(
    #     "{} samples selected (over {})".format(
    #         np.count_nonzero(train_gt), np.count_nonzero(gt)
    #     )
    # )
    # print(
    #     "Running an experiment with the {} model".format(MODEL),
    #     "run {}/{}".format(run + 1, N_RUNS),
    # )

    display_predictions(convert_to_color(gt), viz, caption="Train ground truth")
    # display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    display_predictions(convert_to_color(gt_t), viz, caption="Target Train ground truth")
    # display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SGD":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight, learning_rate="optimal", tol=1e-3, average=10
        )
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "nearest":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = neighbors.KNeighborsClassifier(weights="distance")
        clf = sklearn.model_selection.GridSearchCV(
            clf, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    else:
        random.seed(hyperparams["seed"])
        torch.manual_seed(0)
        cudnn.deterministic = True
        cudnn.benchmark = True
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams["weights"] = torch.from_numpy(weights).cuda().float()
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

        domain_discri1=DomainDiscriminator(model.features_size, hidden_size=128)
        domain_adv1=DomainAdversarialLoss(domain_discri1)
        domain_discri2=DomainDiscriminator(model.features_size*N_CLASSES, hidden_size=128)
        domain_adv2=ConditionalDomainAdversarialLoss(domain_discri2, entropy_conditioning=True,
                    num_classes=N_CLASSES, features_dim=model.features_size, 
                    )        
        # lr = hyperparams.setdefault("learning_rate", 0.01)
        # optimizer = optim.SGD([
        #         {'params': model.parameters()},
        #         {'params': domain_discri1.parameters()},
        #         {'params': domain_discri2.parameters()},
        #     ], lr=lr, weight_decay=0.0005)
        hyperparams.setdefault("batch_size", 100)
        loss = nn.CrossEntropyLoss(reduction='none',weight=hyperparams["weights"],label_smoothing=0.1)
        # loss=CrossEntropyLabelSmooth(reduction='none', num_classes=N_CLASSES,
        #                                              epsilon=0.1,weight=hyperparams["weights"])
        # # Split train set in train/val
        # lr_scheduler    = LambdaLR(optimizer, lambda x: args.lr *
        #                         (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
        # lr_scheduler_ad = LambdaLR(
        # ad_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
        # 

        # Generate the dataset
        hyperparams.update({'flip_augmentation': FLIP_AUGMENTATION, 'radiation_augmentation': RADIATION_AUGMENTATION, 'mixture_augmentation': MIXTURE_AUGMENTATION})
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            # pin_memory=hyperparams['device'],
            shuffle=True,
            drop_last=True
        )

        train_dataset_t = HyperX(img_t, train_t_gt, **hyperparams)
        train_loader_t = data.DataLoader(
            train_dataset_t,
            shuffle=True,
            # pin_memory=hyperparams['device'],
            batch_size=hyperparams["batch_size"],
            drop_last=True
        )     
        # hyperparams.update({'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False})

        # train_dataset_t_weak = HyperX(img_t, train_t_gt, **hyperparams)
        # train_loader_t_weak = data.DataLoader(
        #     train_dataset_t,
        #     shuffle=True,
        #     # pin_memory=hyperparams['device'],
        #     batch_size=hyperparams["batch_size"],
        #     drop_last=True
        # )   
        
        
        hyperparams.update({'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False})
        val_dataset = HyperX(img_t, val_gt, **hyperparams)
        val_loader = data.DataLoader(
            val_dataset,
            # pin_memory=hyperparams['device'],
            batch_size=hyperparams["batch_size"],
        )
        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for _,input, _ ,_,_,_ in train_loader:
                break
            summary(model.to(hyperparams["device"]), input.size()[1:])
            # We would like to use device=hyperparams['device'] altough we have
            # to wait for torchsummary to be fixed first.

        # if CHECKPOINT is not None:
        #     s_model = torch.load(CHECKPOINT)
        #     model_dict = model.state_dict()
        #     state_dict = {k:v for k,v in s_model.items() if k in model_dict.keys()}
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)
        #     #model.load_state_dict(torch.load(CHECKPOINT))
    
        
        Align_dict={"mmd":MMD_loss(kernel_type='rbf'),
                    "mcc":MinimumClassConfusionLoss(temperature=2),
                    "dann":domain_adv1,
                    "cdan":domain_adv2}

        try:
            bestval,mybestval,preds=train(
                model,
                Align_dict,
                optimizer,
                loss,
                train_loader,
                train_loader_t,
                hyperparams["epoch"],
                scheduler=hyperparams["scheduler"],
                device=hyperparams["device"],
                supervision=hyperparams["supervision"],
                hyperparams=hyperparams,
                val_loader=val_loader,
                display=viz,
                m_class=N_CLASSES
            )
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        probabilities = test(model, img_t, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        prediction_na= Mytest(img_t,preds)
        
    acc_s, f_s,y_s = val(model, train_loader,hyperparams["device"],hyperparams["supervision"])
    acc_t, f_t,y_t = val(model, val_loader, hyperparams["device"],hyperparams["supervision"])
    print("acc_s",acc_s,"acc_t",acc_t)
    # tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
    num = np.minimum(len(f_s),len(f_t))
    visualize(f_s.cpu().detach()[0:num], f_t.cpu().detach()[0:num],y=torch.cat((y_s[0:num],y_t[0:num]),0),label=list(range(1, N_CLASSES + 1)),name=DATASET)
 
    run_results = metrics(
        prediction,
        val_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )
    
    run_results_na = metrics(
        prediction_na,
        val_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )
    mask = np.zeros(gt_t.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt_t == l] = True
    prediction[mask] = 0
    prediction_na[mask]=0

    color_prediction = convert_to_color(prediction)
    color_prediction_na = convert_to_color(prediction_na)

    display_predictions(
        color_prediction,
        viz,
        gt=convert_to_color(val_gt),
        caption="Prediction vs. test ground truth",
    )

    display_predictions(
        color_prediction_na,
        viz,
        gt=convert_to_color(val_gt),
        caption="Prediction vs. na test ground truth",
    )
    results.append(run_results)
    results_na.append(run_results_na)

    show_results(run_results, viz, label_values=LABEL_VALUES)
    show_results(run_results_na, viz, label_values=LABEL_VALUES)

    print("bestval",bestval,"mybestval",mybestval)
if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
    show_results(results_na, viz, label_values=LABEL_VALUES, agregated=True)
