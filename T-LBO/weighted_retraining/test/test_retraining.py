# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


# from pathlib import Path
# from torch.nn.functional import binary_cross_entropy


# ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
# sys.path[0] = ROOT_PROJECT
# print('ROOT_PROJECT', ROOT_PROJECT)

# ==========================================================================================
# Test if points with high objective value are close to each other in train & validation set
# ==========================================================================================
# ==========================================================================================
# Setup Data
# ==========================================================================================
# tkwargs = dict(device=torch.device('cuda:1'), dtype=torch.float)
# task = "expr"
# data_folder = f'./weighted_retraining/test/{task}/data'
# plot_folder = f'./weighted_retraining/test/{task}/plots'
# if not os.path.exists(plot_folder):
#     os.makedirs(plot_folder)
# datamodule = torch.load(os.path.join(data_folder, f"datamodule.pt"))
# one_hot_train = torch.from_numpy(datamodule.data_train).to(**tkwargs)
# one_hot_val = torch.from_numpy(datamodule.data_val).to(**tkwargs)
# y_train = torch.from_numpy(datamodule.prop_train).to(**tkwargs)
# y_val = torch.from_numpy(datamodule.prop_val).to(**tkwargs)
#
#
# # compute distances (in BCE metric)
# y_train_std = y_train.add(-y_train.mean()).div(y_train.std())
# targets_train = -y_train
# targets_train_sorted, targets_train_sorted_idx = targets_train.sort()
#
# y_val_std = y_val.add(-y_train.mean()).div(y_train.std())
# targets_val = -y_val
# targets_val_sorted, targets_val_sorted_idx = targets_val.sort()
#
#
# def bce_dist_between_two_points(x, y):
#     assert x.shape == y.shape
#     bce = 0
#     for row in range(x.shape[0]):
#         bce += binary_cross_entropy(x[row], y[row])
#     return bce / x.shape[0]
#
#
# def mean_bce_dist_to_ref_point(ref_point, points):
#     bce_list = []
#     for i in range(points.shape[0]):
#         bce_list.append(bce_dist_between_two_points(ref_point, points[i]))
#     return torch.stack(bce_list).detach().cpu().numpy()
#
#
# N = 500
# best_train_points = one_hot_train[targets_train_sorted_idx[-N:]]
# worst_train_points = one_hot_train[targets_train_sorted_idx[:N]]
# mean_bce_dist_best_train_to_best_val = []
# mean_bce_dist_best_train_to_worst_val = []
# mean_bce_dist_worst_train_to_best_val = []
# mean_bce_dist_worst_train_to_worst_val = []
# i = 0
# for ref_point in best_train_points:
#     print(i)
#     i += 1
#     bce_best_np = mean_bce_dist_to_ref_point(ref_point, one_hot_val[targets_val_sorted_idx][-N:])
#     bce_worst_np = mean_bce_dist_to_ref_point(ref_point, one_hot_val[targets_val_sorted_idx][:N])
#     mean_bce_dist_best_train_to_best_val.append(bce_best_np.mean())
#     mean_bce_dist_best_train_to_worst_val.append(bce_worst_np.mean())
# i = 0
# for ref_point in worst_train_points:
#     print(i)
#     i += 1
#     bce_best_np = mean_bce_dist_to_ref_point(ref_point, one_hot_val[targets_val_sorted_idx][-N:])
#     bce_worst_np = mean_bce_dist_to_ref_point(ref_point, one_hot_val[targets_val_sorted_idx][:N])
#     mean_bce_dist_worst_train_to_best_val.append(bce_best_np.mean())
#     mean_bce_dist_worst_train_to_worst_val.append(bce_worst_np.mean())
#
# mean_bce_dist_best_train_to_best_val = np.array(mean_bce_dist_best_train_to_best_val)
# mean_bce_dist_best_train_to_worst_val = np.array(mean_bce_dist_best_train_to_worst_val)
# mean_bce_dist_worst_train_to_best_val = np.array(mean_bce_dist_worst_train_to_best_val)
# mean_bce_dist_worst_train_to_worst_val = np.array(mean_bce_dist_worst_train_to_worst_val)
#
# plt.scatter(np.arange(N), mean_bce_dist_best_train_to_best_val, label='best train to best val')
# plt.scatter(np.arange(N), mean_bce_dist_best_train_to_worst_val, label='best train to worst val')
# plt.scatter(np.arange(N), mean_bce_dist_worst_train_to_best_val, label='worst train to best val')
# plt.scatter(np.arange(N), mean_bce_dist_worst_train_to_worst_val, label='worst train to worst val')
# plt.legend()
# plt.savefig(os.path.join(data_folder, "bce.pdf"))
# plt.close()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_best = targets_train_sorted[-N:].cpu().numpy()
# x_worst = targets_train_sorted[:N].cpu().numpy()
# y_best = targets_val_sorted[-N:].cpu().numpy()
# y_worst = targets_val_sorted[:N].cpu().numpy()
# ax.scatter(x_best, y_best, mean_bce_dist_best_train_to_best_val, label=f'{N} best train to {N} best val')
# ax.scatter(x_best, y_worst, mean_bce_dist_best_train_to_worst_val, label=f'{N} best train to {N} worst val')
# ax.scatter(x_worst, y_best, mean_bce_dist_worst_train_to_best_val, label=f'{N} worst train to {N} best val')
# ax.scatter(x_worst, y_worst, mean_bce_dist_worst_train_to_worst_val, label=f'{N} worst train to {N} worst val')
# plt.legend()
# plt.xlabel('y-value in train set')
# plt.ylabel('y-value in validation set')
# plt.title('Average bce distance')
# plt.savefig(os.path.join(data_folder, "bce3d.pdf"))
# plt.close()


# ==========================================
# Same but in the latent space with L2 norm
# ==========================================
def l2_dist_between_two_points(x, y):
    y = y.squeeze()
    assert x.shape == y.shape, (x.shape, y.shape)
    return torch.norm(x - y)


def l1_dist_between_two_points(x, y):
    y = y.squeeze()
    assert x.shape == y.shape, (x.shape, y.shape)
    return torch.norm(x - y, p=1)


def cos_dist_between_two_points(x, y):
    y = y.squeeze()
    assert x.shape == y.shape, (x.shape, y.shape)
    return torch.dot(x, y)


def mean_l2_dist_to_ref_point(ref_point, points):
    l2_list = []
    for i in range(points.shape[0]):
        l2_list.append(l2_dist_between_two_points(ref_point, points[i]))
    return torch.stack(l2_list).mean().detach().cpu().numpy()


def mean_l1_dist_to_ref_point(ref_point, points):
    l1_list = []
    for i in range(points.shape[0]):
        l1_list.append(l1_dist_between_two_points(ref_point, points[i]))
    return torch.stack(l1_list).mean().detach().cpu().numpy()


def mean_cos_dist_to_ref_point(ref_point, points):
    cos_list = []
    for i in range(points.shape[0]):
        cos_list.append(cos_dist_between_two_points(ref_point, points[i]))
    return torch.stack(cos_list).mean().detach().cpu().numpy()


def mean_dist_to_ref_point(ref_point, points, dist="l2"):
    if dist == "l2":
        return mean_l2_dist_to_ref_point(ref_point, points)
    elif dist == "l1":
        return mean_l1_dist_to_ref_point(ref_point, points)
    else:
        return mean_cos_dist_to_ref_point(ref_point, points)


# name = "_z-25"
# name = "_z-25_contrastive"
# name = "_z-25_triplet"
# name = "_z-10"
# name = "_z-10_contrastive"
# name = "_z-10_triplet"
# name = "_z-5"
# name = "_z-5_contrastive"
# name = "_z-5_triplet"

# dist = "l2"
#
# x_train = torch.load(os.path.join(data_folder, f"x_train{name}.pt")).cpu()
# y_train = torch.load(os.path.join(data_folder, f"y_train_std{name}.pt")).cpu()
# x_val = torch.load(os.path.join(data_folder, f"x_val{name}.pt")).cpu()
# y_val = torch.load(os.path.join(data_folder, f"y_val_std{name}.pt")).cpu()


def make_latent_space_distance_plots(x_train, y_train, x_val, y_val, plot_folder, name, N=500, dist="l2"):
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    targets_train = -y_train.flatten()
    targets_train_sorted, targets_train_sorted_idx = targets_train.sort()

    targets_val = -y_val
    targets_val_sorted, targets_val_sorted_idx = targets_val.sort()

    # plt.hist(targets_train.cpu().numpy(), bins=100, label='y_train')
    # plt.hist(targets_val.cpu().numpy(), bins=100, label='y_val')
    # plt.legend()
    # plt.savefig(os.path.join(data_folder, f"y_train_val{name}.pdf"))
    # plt.close()

    best_train_points = x_train[targets_train_sorted_idx[-N:]]
    worst_train_points = x_train[targets_train_sorted_idx[:N]]
    best_val_points = x_val[targets_val_sorted_idx][-N:]
    worst_val_points = x_val[targets_val_sorted_idx][:N]

    mean_dist_best_train_to_best_train = []
    mean_dist_best_train_to_worst_train = []
    mean_dist_best_train_to_best_val = []
    mean_dist_best_train_to_worst_val = []
    mean_dist_worst_train_to_best_train = []
    mean_dist_worst_train_to_worst_train = []
    mean_dist_worst_train_to_best_val = []
    mean_dist_worst_train_to_worst_val = []

    mean_dist_best_val_to_best_train = []
    mean_dist_best_val_to_worst_train = []
    mean_dist_best_val_to_best_val = []
    mean_dist_best_val_to_worst_val = []
    mean_dist_worst_val_to_best_train = []
    mean_dist_worst_val_to_worst_train = []
    mean_dist_worst_val_to_best_val = []
    mean_dist_worst_val_to_worst_val = []
    #
    for ref_point in best_train_points:
        best_train_to_best_train = mean_dist_to_ref_point(ref_point, best_train_points, dist=dist)
        best_train_to_worst_train = mean_dist_to_ref_point(ref_point, worst_train_points, dist=dist)
        best_train_to_best_val = mean_dist_to_ref_point(ref_point, best_val_points, dist=dist)
        best_train_to_worst_val = mean_dist_to_ref_point(ref_point, worst_val_points, dist=dist)
        mean_dist_best_train_to_best_train.append(best_train_to_best_train)
        mean_dist_best_train_to_worst_train.append(best_train_to_worst_train)
        mean_dist_best_train_to_best_val.append(best_train_to_best_val)
        mean_dist_best_train_to_worst_val.append(best_train_to_worst_val)
    print("Done")
    for ref_point in worst_train_points:
        worst_train_to_worst_train = mean_dist_to_ref_point(ref_point, worst_train_points, dist=dist)
        worst_train_to_best_train = mean_dist_to_ref_point(ref_point, best_train_points, dist=dist)
        worst_train_to_best_val = mean_dist_to_ref_point(ref_point, best_val_points, dist=dist)
        worst_train_to_worst_val = mean_dist_to_ref_point(ref_point, worst_val_points, dist=dist)
        mean_dist_worst_train_to_best_train.append(worst_train_to_best_train)
        mean_dist_worst_train_to_worst_train.append(worst_train_to_worst_train)
        mean_dist_worst_train_to_best_val.append(worst_train_to_best_val)
        mean_dist_worst_train_to_worst_val.append(worst_train_to_worst_val)
    print("Done")

    for ref_point in best_val_points:
        best_val_to_best_train = mean_dist_to_ref_point(ref_point, best_train_points, dist=dist)
        best_val_to_worst_train = mean_dist_to_ref_point(ref_point, worst_train_points, dist=dist)
        best_val_to_best_val = mean_dist_to_ref_point(ref_point, best_val_points, dist=dist)
        best_val_to_worst_val = mean_dist_to_ref_point(ref_point, worst_val_points, dist=dist)
        mean_dist_best_val_to_best_train.append(best_val_to_best_train)
        mean_dist_best_val_to_worst_train.append(best_val_to_worst_train)
        mean_dist_best_val_to_best_val.append(best_val_to_best_val)
        mean_dist_best_val_to_worst_val.append(best_val_to_worst_val)
    print("Done")
    for ref_point in worst_val_points:
        worst_val_to_worst_train = mean_dist_to_ref_point(ref_point, worst_train_points, dist=dist)
        worst_val_to_best_train = mean_dist_to_ref_point(ref_point, best_train_points, dist=dist)
        worst_val_to_best_val = mean_dist_to_ref_point(ref_point, best_val_points, dist=dist)
        worst_val_to_worst_val = mean_dist_to_ref_point(ref_point, worst_val_points, dist=dist)
        mean_dist_worst_val_to_best_train.append(worst_val_to_worst_train)
        mean_dist_worst_val_to_worst_train.append(worst_val_to_best_train)
        mean_dist_worst_val_to_best_val.append(worst_val_to_best_val)
        mean_dist_worst_val_to_worst_val.append(worst_val_to_worst_val)
    print("Done")

    mean_dist_best_train_to_best_train = np.array(mean_dist_best_train_to_best_train)
    mean_dist_best_train_to_worst_train = np.array(mean_dist_best_train_to_worst_train)
    mean_dist_best_train_to_best_val = np.array(mean_dist_best_train_to_best_val)
    mean_dist_best_train_to_worst_val = np.array(mean_dist_best_train_to_worst_val)
    mean_dist_worst_train_to_best_train = np.array(mean_dist_worst_train_to_best_train)
    mean_dist_worst_train_to_worst_train = np.array(mean_dist_worst_train_to_worst_train)
    mean_dist_worst_train_to_best_val = np.array(mean_dist_worst_train_to_best_val)
    mean_dist_worst_train_to_worst_val = np.array(mean_dist_worst_train_to_worst_val)

    mean_dist_best_val_to_best_train = np.array(mean_dist_best_val_to_best_train)
    mean_dist_best_val_to_worst_train = np.array(mean_dist_best_val_to_worst_train)
    mean_dist_best_val_to_best_val = np.array(mean_dist_best_val_to_best_val)
    mean_dist_best_val_to_worst_val = np.array(mean_dist_best_val_to_worst_val)
    mean_dist_worst_val_to_best_train = np.array(mean_dist_worst_val_to_best_train)
    mean_dist_worst_val_to_worst_train = np.array(mean_dist_worst_val_to_worst_train)
    mean_dist_worst_val_to_best_val = np.array(mean_dist_worst_val_to_best_val)
    mean_dist_worst_val_to_worst_val = np.array(mean_dist_worst_val_to_worst_val)

    all = np.concatenate([
        mean_dist_best_train_to_best_train,
        mean_dist_best_train_to_worst_train,
        mean_dist_best_train_to_best_val,
        mean_dist_best_train_to_worst_val,
        mean_dist_worst_train_to_best_train,
        mean_dist_worst_train_to_worst_train,
        mean_dist_worst_train_to_best_val,
        mean_dist_worst_train_to_worst_val,
        mean_dist_best_val_to_best_train,
        mean_dist_best_val_to_worst_train,
        mean_dist_best_val_to_best_val,
        mean_dist_best_val_to_worst_val,
        mean_dist_worst_val_to_best_train,
        mean_dist_worst_val_to_worst_train,
        mean_dist_worst_val_to_best_val,
        mean_dist_worst_val_to_worst_val
    ])
    all_min = all.min()
    all_max = all.max()

    plt.scatter(np.arange(N), mean_dist_best_train_to_best_val,
                label=f"best train to best val ({mean_dist_best_train_to_best_val.mean():.3f}"
                      f"$\pm${mean_dist_best_train_to_best_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(N), mean_dist_best_train_to_worst_val,
                label=f"best train to worst val ({mean_dist_best_train_to_worst_val.mean():.3f}"
                      f"$\pm${mean_dist_best_train_to_worst_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(N), mean_dist_worst_train_to_best_val,
                label=f"worst train to best val ({mean_dist_worst_train_to_best_val.mean():.3f}"
                      f"$\pm${mean_dist_worst_train_to_best_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(N), mean_dist_worst_train_to_worst_val,
                label=f"worst train to worst val ({mean_dist_worst_train_to_worst_val.mean():.3f}"
                      f"$\pm${mean_dist_worst_train_to_worst_val.std():.3f})",
                alpha=0.25)
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"train_to_val_{dist}_{name}.pdf"))
    plt.close()

    plt.hist(mean_dist_best_train_to_best_val,
             label=f"best train to best val ({mean_dist_best_train_to_best_val.mean():.3f}"
                   f"$\pm${mean_dist_best_train_to_best_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_best_train_to_worst_val,
             label=f"best train to worst val ({mean_dist_best_train_to_worst_val.mean():.3f}"
                   f"$\pm${mean_dist_best_train_to_worst_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_train_to_best_val,
             label=f"worst train to best val ({mean_dist_worst_train_to_best_val.mean():.3f}"
                   f"$\pm${mean_dist_worst_train_to_best_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_train_to_worst_val,
             label=f"worst train to worst val ({mean_dist_worst_train_to_worst_val.mean():.3f}"
                   f"$\pm${mean_dist_worst_train_to_worst_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"train_to_val_{dist}_{name}_hist.pdf"))
    plt.close()

    plt.scatter(np.arange(len(mean_dist_best_val_to_best_train)), mean_dist_best_val_to_best_train,
                label=f"best val to best train ({mean_dist_best_val_to_best_train.mean():.3f}"
                      f"$\pm${mean_dist_best_val_to_best_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_best_val_to_worst_train)), mean_dist_best_val_to_worst_train,
                label=f"best val to worst train ({mean_dist_best_val_to_worst_train.mean():.3f}"
                      f"$\pm${mean_dist_best_val_to_worst_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_val_to_best_train)), mean_dist_worst_val_to_best_train,
                label=f"worst val to best train ({mean_dist_worst_val_to_best_train.mean():.3f}"
                      f"$\pm${mean_dist_worst_val_to_best_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_val_to_worst_train)), mean_dist_worst_val_to_worst_train,
                label=f"worst val to worst train ({mean_dist_worst_val_to_worst_train.mean():.3f}"
                      f"$\pm${mean_dist_worst_val_to_worst_train.std():.3f})",
                alpha=0.25)
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"val_to_train_{dist}_{name}.pdf"))
    plt.close()

    plt.hist(mean_dist_best_val_to_best_train,
             label=f"best val to best train ({mean_dist_best_val_to_best_train.mean():.3f}"
                   f"$\pm${mean_dist_best_val_to_best_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_best_val_to_worst_train,
             label=f"best val to worst train ({mean_dist_best_val_to_worst_train.mean():.3f}"
                   f"$\pm${mean_dist_best_val_to_worst_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_val_to_best_train,
             label=f"worst val to best train ({mean_dist_worst_val_to_best_train.mean():.3f}"
                   f"$\pm${mean_dist_worst_val_to_best_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_val_to_worst_train,
             label=f"worst val to worst train ({mean_dist_worst_val_to_worst_train.mean():.3f}"
                   f"$\pm${mean_dist_worst_val_to_worst_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"val_to_train_{dist}_{name}_hist.pdf"))
    plt.close()

    plt.scatter(np.arange(len(mean_dist_best_train_to_best_train)), mean_dist_best_train_to_best_train,
                label=f"best train to best train ({mean_dist_best_train_to_best_train.mean():.3f}"
                      f"$\pm${mean_dist_best_train_to_best_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_best_train_to_worst_train)), mean_dist_best_train_to_worst_train,
                label=f"best train to worst train ({mean_dist_best_train_to_worst_train.mean():.3f}"
                      f"$\pm${mean_dist_best_train_to_worst_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_train_to_best_train)), mean_dist_worst_train_to_best_train,
                label=f"worst train to best train ({mean_dist_worst_train_to_best_train.mean():.3f}"
                      f"$\pm${mean_dist_worst_train_to_best_train.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_train_to_worst_train)), mean_dist_worst_train_to_worst_train,
                label=f"worst train to worst train ({mean_dist_worst_train_to_worst_train.mean():.3f}"
                      f"$\pm${mean_dist_worst_train_to_worst_train.std():.3f})",
                alpha=0.25)
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"train_to_train_{dist}_{name}.pdf"))
    plt.close()

    plt.hist(mean_dist_best_train_to_best_train,
             label=f"best train to best train ({mean_dist_best_train_to_best_train.mean():.3f}"
                   f"$\pm${mean_dist_best_train_to_best_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_best_train_to_worst_train,
             label=f"best train to worst train ({mean_dist_best_train_to_worst_train.mean():.3f}"
                   f"$\pm${mean_dist_best_train_to_worst_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_train_to_best_train,
             label=f"worst train to best train ({mean_dist_worst_train_to_best_train.mean():.3f}"
                   f"$\pm${mean_dist_worst_train_to_best_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_train_to_worst_train,
             label=f"worst train to worst train ({mean_dist_worst_train_to_worst_train.mean():.3f}"
                   f"$\pm${mean_dist_worst_train_to_worst_train.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"train_to_train_{dist}_{name}_hist.pdf"))
    plt.close()

    plt.scatter(np.arange(len(mean_dist_best_val_to_best_val)), mean_dist_best_val_to_best_val,
                label=f"best val to best val ({mean_dist_best_val_to_best_val.mean():.3f}"
                      f"$\pm${mean_dist_best_val_to_best_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_best_val_to_worst_val)), mean_dist_best_val_to_worst_val,
                label=f"best val to worst val ({mean_dist_best_val_to_worst_val.mean():.3f}"
                      f"$\pm${mean_dist_best_val_to_worst_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_val_to_best_val)), mean_dist_worst_val_to_best_val,
                label=f"worst val to best val ({mean_dist_worst_val_to_best_val.mean():.3f}"
                      f"$\pm${mean_dist_worst_val_to_best_val.std():.3f})",
                alpha=0.25)
    plt.scatter(np.arange(len(mean_dist_worst_val_to_worst_val)), mean_dist_worst_val_to_worst_val,
                label=f"worst val to worst val ({mean_dist_worst_val_to_worst_val.mean():.3f}"
                      f"$\pm${mean_dist_worst_val_to_worst_val.std():.3f})",
                alpha=0.25)
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"val_to_val_{dist}_{name}.pdf"))
    plt.close()

    plt.hist(mean_dist_best_val_to_best_val,
             label=f"best val to best val ({mean_dist_best_val_to_best_val.mean():.3f}"
                   f"$\pm${mean_dist_best_val_to_best_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_best_val_to_worst_val,
             label=f"best val to worst val ({mean_dist_best_val_to_worst_val.mean():.3f}"
                   f"$\pm${mean_dist_best_val_to_worst_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_val_to_best_val,
             label=f"worst val to best val ({mean_dist_worst_val_to_best_val.mean():.3f}"
                   f"$\pm${mean_dist_worst_val_to_best_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.hist(mean_dist_worst_val_to_worst_val,
             label=f"worst val to worst val ({mean_dist_worst_val_to_worst_val.mean():.3f}"
                   f"$\pm${mean_dist_worst_val_to_worst_val.std():.3f})",
             alpha=0.25, bins=50, range=[all_min, all_max])
    plt.legend()
    plt.title(f"Mean {dist} distance in latent space")
    plt.savefig(os.path.join(plot_folder, f"val_to_val_{dist}_{name}_hist.pdf"))
    plt.close()

    # train_best = targets_train_sorted[-N:].cpu().numpy()
    # train_worst = targets_train_sorted[:N].cpu().numpy()
    # val_best = targets_val_sorted[-N:].cpu().numpy()
    # val_worst = targets_val_sorted[:N].cpu().numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_best, train_best, mean_dist_best_train_to_best_train, label=f'{N} best train to {N} best train', alpha=0.25)
    # ax.scatter(train_best, train_worst, mean_dist_best_train_to_worst_val, label=f'{N} best train to {N} worst train', alpha=0.25)
    # ax.scatter(train_worst, train_best, mean_dist_worst_train_to_best_val, label=f'{N} worst train to {N} best train', alpha=0.25)
    # ax.scatter(train_worst, train_worst, mean_dist_worst_train_to_worst_val, label=f'{N} worst train to {N} worst train', alpha=0.25)
    # plt.legend()
    # plt.xlabel('y-value in train set')
    # plt.ylabel('y-value in train set')
    # plt.title(f'Average {dist} distance (in latent space)')
    # plt.savefig(os.path.join(plot_folder, f"3d_train_to_train_{dist}{name}.pdf"))
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(val_best, val_best, mean_dist_best_train_to_best_train, label=f'{N} best val to {N} best val', alpha=0.25)
    # ax.scatter(val_best, val_worst, mean_dist_best_train_to_worst_val, label=f'{N} best val to {N} worst val', alpha=0.25)
    # ax.scatter(val_worst, val_best, mean_dist_worst_train_to_best_val, label=f'{N} worst val to {N} best val', alpha=0.25)
    # ax.scatter(val_worst, val_worst, mean_dist_worst_train_to_worst_val, label=f'{N} worst val to {N} worst val', alpha=0.25)
    # plt.legend()
    # plt.xlabel('y-value in val set')
    # plt.ylabel('y-value in val set')
    # plt.title(f'Average {dist} distance (in latent space)')
    # plt.savefig(os.path.join(plot_folder, f"3d_val_to_val_{dist}{name}.pdf"))
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_best, val_best, mean_dist_best_train_to_best_train, label=f'{N} best train to {N} best val', alpha=0.25)
    # ax.scatter(train_best, val_worst, mean_dist_best_train_to_worst_val, label=f'{N} best train to {N} worst val', alpha=0.25)
    # ax.scatter(train_worst, val_best, mean_dist_worst_train_to_best_val, label=f'{N} worst train to {N} best val', alpha=0.25)
    # ax.scatter(train_worst, val_worst, mean_dist_worst_train_to_worst_val, label=f'{N} worst train to {N} worst val', alpha=0.25)
    # plt.legend()
    # plt.xlabel('y-value in train set')
    # plt.ylabel('y-value in val set')
    # plt.title(f'Average {dist} distance (in latent space)')
    # plt.savefig(os.path.join(plot_folder, f"3d_train_to_val_{dist}{name}.pdf"))
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # train_best = targets_train_sorted[-N:].cpu().numpy()
    # train_worst = targets_train_sorted[:N].cpu().numpy()
    # val_best = targets_val_sorted[-N:].cpu().numpy()
    # val_worst = targets_val_sorted[:N].cpu().numpy()
    # ax.scatter(train_best, val_best, mean_dist_best_train_to_best_val, label=f'{N} best train to {N} best val')
    # ax.scatter(train_best, val_worst, mean_dist_best_train_to_worst_val, label=f'{N} best train to {N} worst val')
    # ax.scatter(train_worst, val_best, mean_dist_worst_train_to_best_val, label=f'{N} worst train to {N} best val')
    # ax.scatter(train_worst, val_worst, mean_dist_worst_train_to_worst_val, label=f'{N} worst train to {N} worst val')
    # plt.legend()
    # plt.xlabel('y-value in train set')
    # plt.ylabel('y-value in validation set')
    # plt.title('Average L2 distance (in latent space)')
    # plt.savefig(os.path.join(plot_folder, f"3d{name}.pdf"))
    # plt.close()
    #
    # if x_train.size(1) == 2:
    #     plt.scatter(best_train_points[:, 0], best_train_points[:, 1], label=f'{N} best train', alpha=0.25)
    #     plt.scatter(worst_train_points[:, 0], worst_train_points[:, 1], label=f'{N} best worst', alpha=0.25)
    #     plt.scatter(best_val_points[:, 0], best_val_points[:, 1], label=f'{N} best val', alpha=0.25)
    #     plt.scatter(worst_val_points[:, 0], worst_val_points[:, 1], label=f'{N} worst val', alpha=0.25)
    #     plt.title("2D Latent Space")
    #     plt.legend()
    #     plt.savefig(os.path.join(plot_folder, f"space_2d{name}.pdf"))
    #     plt.close()
    #
    # if x_train.size(1) == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(best_train_points[:, 0], best_train_points[:, 1], best_train_points[:, 2], label=f'{N} best train', alpha=0.25, marker='.')
    #     ax.scatter(worst_train_points[:, 0], worst_train_points[:, 1], worst_train_points[:, 2], label=f'{N} best worst', alpha=0.25, marker='.')
    #     ax.scatter(best_val_points[:, 0], best_val_points[:, 1], best_val_points[:, 2], label=f'{N} best val', alpha=0.25, marker='*')
    #     ax.scatter(worst_val_points[:, 0], worst_val_points[:, 1], worst_val_points[:, 2], label=f'{N} worst val', alpha=0.25, marker='*')
    #     plt.title("3D Latent Space")
    #     plt.legend()
    #     plt.savefig(os.path.join(plot_folder, f"space_3d{name}.pdf"))
    #     plt.close()
    #
    # for i in range(5):
    #     reducer = umap.UMAP()
    #     best_train_embedding = reducer.fit_transform(best_train_points.cpu())
    #     worst_train_embedding = reducer.fit_transform(worst_train_points.cpu())
    #     best_val_embedding = reducer.fit_transform(best_val_points.cpu())
    #     worst_val_embedding = reducer.fit_transform(worst_val_points.cpu())
    #     plt.scatter(best_train_embedding[:, 0], best_train_embedding[:, 1], label=f'best {N} from train set', marker='.')
    #     plt.scatter(worst_train_embedding[:, 0], worst_train_embedding[:, 1], label=f'worst {N} from train set', marker='.')
    #     plt.scatter(best_val_embedding[:, 0], best_val_embedding[:, 1], label=f'best {N} from val set', marker='.')
    #     plt.scatter(worst_val_embedding[:, 0], worst_val_embedding[:, 1], label=f'worst {N} from val set', marker='.')
    #     plt.legend()
    #     plt.savefig(os.path.join(plot_folder, f'umap{i}{name}.pdf'))
    #     plt.close()

    best_train_features = best_train_points.cpu().numpy()
    worst_train_features = worst_train_points.cpu().numpy()
    best_val_features = best_val_points.cpu().numpy()
    worst_val_features = worst_val_points.cpu().numpy()

    # UMAP 2d and 3d plots
    umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
    umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)

    best_train_projections_2d = umap_2d.fit_transform(best_train_features, )
    worst_train_projections_2d = umap_2d.fit_transform(worst_train_features, )
    best_val_projections_2d = umap_2d.fit_transform(best_val_features, )
    worst_val_projections_2d = umap_2d.fit_transform(worst_val_features, )
    data_2d = np.concatenate([
        best_train_projections_2d,
        worst_train_projections_2d,
        best_val_projections_2d,
        worst_val_projections_2d
    ])
    df_2d = pd.DataFrame(data_2d)
    df_2d['class'] = ["best_train"] * 500 + ["worst_train"] * 500 + ["best_val"] * 500 + ["worst_val"] * 500

    best_train_projections_3d = umap_3d.fit_transform(best_train_features, )
    worst_train_projections_3d = umap_3d.fit_transform(worst_train_features, )
    best_val_projections_3d = umap_3d.fit_transform(best_val_features, )
    worst_val_projections_3d = umap_3d.fit_transform(worst_val_features, )
    data_3d = np.concatenate([
        best_train_projections_3d,
        worst_train_projections_3d,
        best_val_projections_3d,
        worst_val_projections_3d
    ])
    df_3d = pd.DataFrame(data_3d)
    df_3d['class'] = ["best_train"] * 500 + ["worst_train"] * 500 + ["best_val"] * 500 + ["worst_val"] * 500

    fig_2d = px.scatter(
        df_2d, x=0, y=1,
        color=df_2d["class"], labels={'color': 'class'}
    )
    fig_2d.update_traces(marker_size=5)
    fig_2d.write_image(os.path.join(plot_folder, f'umap_2d_{name}.pdf'))
    plt.close()

    fig_3d = px.scatter_3d(
        df_3d, x=0, y=1, z=2,
        color=df_3d['class'], labels={'color': 'class'}
    )
    fig_3d.update_traces(marker_size=3)
    fig_3d.write_image(os.path.join(plot_folder, f'umap_3d_{name}.pdf'))
    plt.close()

    # t-SNE 3d and 2d plots
    tsne_3d = TSNE(n_components=3, random_state=0)
    best_train_projections_3d = tsne_3d.fit_transform(best_train_features, )
    worst_train_projections_3d = tsne_3d.fit_transform(worst_train_features, )
    best_val_projections_3d = tsne_3d.fit_transform(best_val_features, )
    worst_val_projections_3d = tsne_3d.fit_transform(worst_val_features, )
    data_3d = np.concatenate([
        best_train_projections_3d,
        worst_train_projections_3d,
        best_val_projections_3d,
        worst_val_projections_3d
    ])
    df_3d = pd.DataFrame(data_3d)
    df_3d['class'] = ["best_train"] * 500 + ["worst_train"] * 500 + ["best_val"] * 500 + ["worst_val"] * 500
    fig = px.scatter_3d(df_3d, x=0, y=1, z=2, color=df_3d['class'], labels={'color': 'class'})
    fig.update_traces(marker_size=5)
    fig.write_image(os.path.join(plot_folder, f'tsne_3d_{name}.pdf'))
    plt.close()

    tsne_2d = TSNE(n_components=2, random_state=0)
    best_train_projections_2d = tsne_2d.fit_transform(best_train_features, )
    worst_train_projections_2d = tsne_2d.fit_transform(worst_train_features, )
    best_val_projections_2d = tsne_2d.fit_transform(best_val_features, )
    worst_val_projections_2d = tsne_2d.fit_transform(worst_val_features, )
    data_2d = np.concatenate([
        best_train_projections_2d,
        worst_train_projections_2d,
        best_val_projections_2d,
        worst_val_projections_2d
    ])
    df_2d = pd.DataFrame(data_2d)
    df_2d['class'] = ["best_train"] * 500 + ["worst_train"] * 500 + ["best_val"] * 500 + ["worst_val"] * 500
    fig = px.scatter(df_2d, x=0, y=1, color=df_2d['class'], labels={'color': 'class'})
    fig.update_traces(marker_size=5)
    fig.write_image(os.path.join(plot_folder, f'tsne_2d_{name}.pdf'))
    plt.close()
