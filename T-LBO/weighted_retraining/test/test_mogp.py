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
import torch
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from matplotlib import pyplot as plt

from utils.utils_save import ROOT_PROJECT
from weighted_retraining.weighted_retraining.bo_torch.gp_torch import gp_torch_train

# from pathlib import Path
# ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
# sys.path[0] = ROOT_PROJECT
# print('ROOT_PROJECT', ROOT_PROJECT)


# ========================================================================
# Test MOGP model
# ========================================================================
# Setup Data
# ========================================================================
tkwargs = dict(device=torch.device('cuda:1'), dtype=torch.float)
data_folder = os.path.join(ROOT_PROJECT, 'weighted_retraining/test/data/')
name = "_z-5_triplet"

x_train = torch.load(os.path.join(data_folder, f"x_train{name}.pt")).to(**tkwargs)
x_val = torch.load(os.path.join(data_folder, f"x_val{name}.pt")).to(**tkwargs)
# y_train = torch.load(os.path.join(data_folder, f"y_train_normalized{name}.pt")).to(**tkwargs)
# y_train = torch.load(os.path.join(data_folder, f"y_train_std{name}.pt")).to(**tkwargs)
y_train = torch.load(os.path.join(data_folder, f"y_train{name}.pt")).to(**tkwargs)
# y_val = torch.load(os.path.join(data_folder, f"y_val_normalized{name}.pt")).to(**tkwargs)
# y_val = torch.load(os.path.join(data_folder, f"y_val_std{name}.pt")).to(**tkwargs)
y_val = torch.load(os.path.join(data_folder, f"y_val{name}.pt")).to(**tkwargs)
# r_train = torch.load(os.path.join(data_folder, f"r_train_normalized{name}.pt")).to(**tkwargs)
# r_train = torch.load(os.path.join(data_folder, f"r_train_std{name}.pt")).to(**tkwargs)
r_train = torch.load(os.path.join(data_folder, f"r_train{name}.pt")).to(**tkwargs)
# r_val = torch.load(os.path.join(data_folder, f"r_val_normalized{name}.pt")).to(**tkwargs)
# r_val = torch.load(os.path.join(data_folder, f"r_val_std{name}.pt")).to(**tkwargs)
r_val = torch.load(os.path.join(data_folder, f"r_val{name}.pt")).to(**tkwargs)

df_mse_y = pd.DataFrame(columns=['Model Configuration', 'Train MSE', 'Validation MSE'])
df_mse_r = pd.DataFrame(columns=['Model Configuration', 'Train MSE', 'Validation MSE'])

TRAIN_GP_RECON = True
target_transform = ["raw", "standardize"]  # ["raw", "standardize", "normalize"]
kernel_names = ["rbf", "matern-52"]  # ['matern-52', 'rbf']
exp_targets = [False]
input_warping = [False]
output_warping = [False, True]
target_subset = [False]

if TRAIN_GP_RECON:
    gp_recon_model_name = ["gp_recon_"]
    for transfo in target_transform:
        gp_recon_model_name.append(transfo + "_")
        for ker_name in kernel_names:
            gp_recon_model_name.append(ker_name + '_')
            for inp_w in input_warping:
                gp_recon_model_name.append("inp-w_" if inp_w else "")
                gp_recon_model_name.append(name)
                # ========================================================================
                # Train Model
                # ========================================================================
                gp_recon_model_name_str = ''.join(gp_recon_model_name)
                print('gp error model name', gp_recon_model_name_str)
                save_folder = os.path.join(data_folder, gp_recon_model_name_str)

                if not os.path.exists(os.path.join(data_folder, gp_recon_model_name_str)):
                    os.mkdir(os.path.join(data_folder, gp_recon_model_name_str))

                r_train_target = r_train.exp() if transfo == "exp" else r_train
                if transfo == 'normalize':
                    rbounds = torch.zeros(2, r_train.shape[1], **tkwargs)
                    rbounds[0] = torch.quantile(r_train, .0005, dim=0)
                    rbounds[1] = torch.quantile(r_train, .9995, dim=0)
                    rdelta = .05 * (rbounds[1] - rbounds[0])
                    rbounds[0] -= rdelta
                    rbounds[1] += rdelta
                    r_train_target = normalize(r_train, rbounds)

                train_kw_error = dict(
                    train_x=x_train,
                    train_y=r_train_target,
                    n_inducing_points=500,
                    tkwargs=tkwargs,
                    init=True,
                    scale=True,
                    covar_name=ker_name,
                    gp_file='',
                    save_file=os.path.join(data_folder, f"{gp_recon_model_name_str}/gp.npz"),
                    input_wp=inp_w,
                    outcome_transform=Standardize(m=1) if transfo == 'standardize' else None,
                    options={'lr': 1e-2, 'maxiter': 500}
                )

                gp_recon_model = gp_torch_train(**train_kw_error)
                gp_recon_model.eval()

                # ========================================================================
                # Test Models
                # ========================================================================
                with torch.no_grad():
                    pred_r_train_mean = gp_recon_model.posterior(x_train).mean
                    pred_r_train_std = gp_recon_model.posterior(x_train).variance.sqrt()
                    pred_r_test_mean = gp_recon_model.posterior(x_val).mean
                    pred_r_test_std = gp_recon_model.posterior(x_val).variance.sqrt()

                    pred_r_train_mean = pred_r_train_mean.clamp_min(
                        1e-4).log() if transfo == "exp" else pred_r_train_mean
                    pred_r_train_std = pred_r_train_std.clamp_min(
                        1e-4).log() if transfo == "exp" else pred_r_train_std
                    pred_r_test_mean = pred_r_test_mean.clamp_min(
                        1e-4).log() if transfo == "exp" else pred_r_test_mean
                    pred_r_test_std = pred_r_test_std.clamp_min(1e-4).log() if transfo == "exp" else pred_r_test_std

                    pred_r_train_mean = unnormalize(pred_r_train_mean,
                                                    rbounds) if transfo == "normalize" else pred_r_train_mean
                    pred_r_train_std = unnormalize(pred_r_train_std,
                                                   rbounds) if transfo == "normalize" else pred_r_train_std
                    pred_r_test_mean = unnormalize(pred_r_test_mean,
                                                   rbounds) if transfo == "normalize" else pred_r_test_mean
                    pred_r_test_std = unnormalize(pred_r_test_std,
                                                  rbounds) if transfo == "normalize" else pred_r_test_std

                    gp_recon_model_fit_train = (pred_r_train_mean - r_train).pow(2).div(len(r_train))
                    gp_recon_model_fit_test = (pred_r_test_mean - r_val).pow(2).div(len(r_train))
                    torch.save(gp_recon_model_fit_train, os.path.join(save_folder, "gp_recon_model_fit_train.pt"))
                    torch.save(gp_recon_model_fit_test, os.path.join(save_folder, "gp_recon_model_fit_test.pt"))
                    print(f'\tMSE on r train set     : {gp_recon_model_fit_train.sum().item():.3f}')
                    print(f'\tMSE on r validation set: {gp_recon_model_fit_test.sum().item():.3f}')

                    df_ = pd.DataFrame(
                        [[gp_recon_model_name_str,
                          gp_recon_model_fit_train.sum().item(),
                          gp_recon_model_fit_test.sum().item()]],
                        columns=df_mse_r.columns)
                    df_mse_r = df_mse_r.append(df_, ignore_index=True)
                    df_mse_r.to_pickle(os.path.join(data_folder, 'mse_r.pkl'))

                    # error plots
                    error_train_sorted, indices_train_pred = torch.sort(r_train, dim=0)
                    gp_r_train_pred_sorted = pred_r_train_mean[indices_train_pred].view(-1, 1)
                    gp_r_train_pred_std_sorted = pred_r_train_std[indices_train_pred].view(-1, 1)
                    plt.scatter(np.arange(len(indices_train_pred)), error_train_sorted.cpu().numpy(),
                                label='err true', marker='+', color='C1', s=15)
                    plt.errorbar(np.arange(len(indices_train_pred)),
                                 gp_r_train_pred_sorted.detach().cpu().numpy().flatten(),
                                 yerr=gp_r_train_pred_std_sorted.detach().cpu().numpy().flatten(),
                                 fmt='*', alpha=0.05, label='err pred', color='C0', ecolor='C0')
                    plt.scatter(np.arange(len(indices_train_pred)),
                                gp_r_train_pred_sorted.detach().cpu().numpy(),
                                marker='*', alpha=0.2, s=10, color='C0')
                    plt.legend()
                    plt.title('error predictions and uncertainty on train set')
                    plt.savefig(os.path.join(save_folder, 'gp_error_train_uncertainty.pdf'))
                    plt.close()

                    error_test_sorted, indices_test_pred = torch.sort(r_val, dim=0)
                    gp_r_test_pred_sorted = pred_r_test_mean[indices_test_pred].view(-1, 1)
                    gp_r_test_pred_std_sorted = pred_r_test_std[indices_test_pred].view(-1, 1)
                    plt.scatter(np.arange(len(indices_test_pred)), error_test_sorted.cpu().numpy(),
                                label='err true',
                                marker='+', color='C1', s=15)
                    plt.errorbar(np.arange(len(indices_test_pred)),
                                 gp_r_test_pred_sorted.detach().cpu().numpy().flatten(),
                                 yerr=gp_r_test_pred_std_sorted.detach().cpu().numpy().flatten(),
                                 marker='*', alpha=0.05, label='err pred', color='C0', ecolor='C0')
                    plt.scatter(np.arange(len(indices_test_pred)),
                                gp_r_test_pred_sorted.detach().cpu().numpy().flatten(),
                                marker='*', color='C0', alpha=0.2, s=10)
                    plt.legend()
                    plt.title('error predictions and uncertainty on test set')
                    plt.savefig(os.path.join(save_folder, 'gp_error_test_uncertainty.pdf'))
                    plt.close()

                    if x_train.size(1) == 2:
                        from itertools import product

                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        ax.scatter(x_train[:, 0].cpu().numpy(), x_train[:, 1].cpu().numpy(),
                                   r_train.cpu().numpy(), label=f"r train true", marker="+",
                                   alpha=0.05, color='C0')
                        ax.scatter(x_val[:, 0].cpu().numpy(), x_val[:, 1].cpu().numpy(),
                                   r_val.cpu().numpy(), label=f"r test true", marker="+",
                                   alpha=0.05, color='C4')
                        X, Y = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
                        X_all = np.array(list(product(X, Y)))
                        X, Y = np.meshgrid(X, Y)
                        pred_r_train_mean_grid = gp_recon_model.posterior(
                            torch.from_numpy(X_all).to(**tkwargs)).mean
                        pred_r_train_std_grid = gp_recon_model.posterior(
                            torch.from_numpy(X_all).to(**tkwargs)).variance.sqrt()
                        Z = pred_r_train_mean_grid.view(100, 100).detach().cpu().numpy()
                        ZZ = pred_r_train_std_grid.view(100, 100).detach().cpu().numpy()
                        ax.plot_surface(X, Y, Z, alpha=0.15, color='C1')
                        # ax.plot_surface(X, Y, Z + ZZ, alpha=0.15, color='C2')
                        # ax.plot_surface(X, Y, Z - ZZ, alpha=0.15, color='C3')
                        plt.title('GP pred recon. err.')
                        plt.savefig(os.path.join(save_folder, f"gp_err_pred_viz_3D{name}.pdf"))
                        plt.close()

                gp_recon_model_name.pop()
                gp_recon_model_name.pop()
            gp_recon_model_name.pop()
            gp_recon_model_name.pop()
        gp_recon_model_name.pop()
else:
    gp_obj_model_name = ["gp_obj_"]
    for transfo in target_transform:
        gp_obj_model_name.append(transfo + "_")
        for ker_name in kernel_names:
            gp_obj_model_name.append(ker_name + '_')
            for exp_target in exp_targets:
                gp_obj_model_name.append("exp-y_" if exp_target else "")
                for inp_w in input_warping:
                    gp_obj_model_name.append("inp-w_" if inp_w else "")
                    for subset in target_subset:
                        gp_obj_model_name.append("subset-y" if subset else "")
                        gp_obj_model_name.append(name)
                        # ========================================================================
                        # Train Model
                        # ========================================================================
                        gp_obj_model_name_str = ''.join(gp_obj_model_name)
                        print('gp model name ', gp_obj_model_name_str)
                        if not os.path.exists(os.path.join(data_folder, gp_obj_model_name_str)):
                            os.mkdir(os.path.join(data_folder, gp_obj_model_name_str))
                        save_folder = os.path.join(data_folder, gp_obj_model_name_str)

                        y_train_target = -y_train  # minimize
                        y_test_target = -y_val  # minimize
                        y_train_target = y_train_target.exp() if exp_target else y_train_target

                        if transfo == 'normalize':
                            bounds = torch.zeros(2, y_train.shape[1], **tkwargs)
                            bounds[0] = torch.quantile(y_train, .0005, dim=0)
                            bounds[1] = torch.quantile(y_train, .9995, dim=0)
                            delta = .05 * (bounds[1] - bounds[0])
                            bounds[0] -= delta
                            bounds[1] += delta
                            y_train_target = normalize(y_train, bounds)

                        if subset:
                            alpha = torch.tensor(0.2).to(**tkwargs)
                            train_quantile = torch.quantile(y_train_target, alpha)
                            keep_train_idx = torch.tensor(y_train_target > train_quantile).view(-1)
                            x_train = x_train[keep_train_idx, :]
                            y_train_target = y_train_target[keep_train_idx]
                            # test_quantile = torch.quantile(y_train_target, alpha)
                            # keep_test_idx = torch.tensor(y_test_target > test_quantile).view(-1)
                            # x_test = x_test[keep_test_idx, :]
                            # y_test_target = y_test_target[keep_test_idx]

                        train_kw = dict(
                            train_x=x_train,
                            train_y=y_train_target,
                            n_inducing_points=500,
                            tkwargs=tkwargs,
                            init=True,
                            scale=True,
                            covar_name=ker_name,
                            gp_file='',
                            save_file=os.path.join(data_folder, f"{gp_obj_model_name_str}/gp.npz"),
                            input_wp=inp_w,
                            outcome_transform=Standardize(m=1) if transfo == "standardize" else None,
                            options={'lr': 0.05, 'maxiter': 100}
                        )

                        gp_obj_model = gp_torch_train(**train_kw)
                        gp_obj_model.eval()

                        # ========================================================================
                        # Test Models
                        # ========================================================================
                        with torch.no_grad():
                            pred_y_train_mean = gp_obj_model.posterior(x_train).mean
                            pred_y_train_std = gp_obj_model.posterior(x_train).variance.sqrt()
                            pred_y_test_mean = gp_obj_model.posterior(x_val).mean
                            pred_y_test_std = gp_obj_model.posterior(x_val).variance.sqrt()

                            pred_y_train_mean = pred_y_train_mean.clamp_min(
                                1e-4).log() if exp_target else pred_y_train_mean
                            pred_y_train_std = pred_y_train_std.clamp_min(
                                1e-4).log() if exp_target else pred_y_train_std
                            pred_y_test_mean = pred_y_test_mean.clamp_min(
                                1e-4).log() if exp_target else pred_y_test_mean
                            pred_y_test_std = pred_y_test_std.clamp_min(
                                1e-4).log() if exp_target else pred_y_test_std

                            pred_r_train_mean = unnormalize(pred_y_train_mean,
                                                            bounds) if transfo == "normalize" else pred_y_train_mean
                            pred_y_train_std = unnormalize(pred_y_train_std,
                                                           bounds) if transfo == "normalize" else pred_y_train_std
                            pred_y_test_mean = unnormalize(pred_y_test_mean,
                                                           bounds) if transfo == "normalize" else pred_y_test_mean
                            pred_y_test_std = unnormalize(pred_y_test_std,
                                                          bounds) if transfo == "normalize" else pred_y_test_std

                            gp_obj_model_fit_train = (pred_y_train_mean + y_train).pow(2).div(len(r_train))
                            gp_obj_model_fit_test = (pred_y_test_mean + y_val).pow(2).div(len(r_train))
                            torch.save(gp_obj_model_fit_train.sum(),
                                       os.path.join(save_folder, f"gp_obj_model_fit_train.pt"))
                            torch.save(gp_obj_model_fit_test.sum(),
                                       os.path.join(save_folder, f"gp_obj_model_fit_test.pt"))
                            print(f'\tMSE on y train set     : {gp_obj_model_fit_train.sum().item():.3f}')
                            print(f'\tMSE on y validation set: {gp_obj_model_fit_test.sum().item():.3f}')

                            df_ = pd.DataFrame(
                                [[gp_obj_model_name_str,
                                  gp_obj_model_fit_train.sum().item(),
                                  gp_obj_model_fit_test.sum().item()]],
                                columns=df_mse_y.columns)

                            df_mse_y = df_mse_y.append(df_, ignore_index=True)
                            df_mse_y.to_pickle(os.path.join(data_folder, 'mse_y.pkl'))

                            # y plots
                            y_train_sorted, indices_train_pred = torch.sort(-y_train, dim=0)
                            gp_y_train_pred_sorted = pred_y_train_mean[indices_train_pred].view(-1, 1)
                            gp_y_train_pred_std_sorted = pred_y_train_std[indices_train_pred].view(-1, 1)
                            plt.scatter(np.arange(len(indices_train_pred)), y_train_sorted.cpu().numpy(),
                                        label='y true', marker='+', color='C1', s=15)
                            plt.errorbar(np.arange(len(indices_train_pred)),
                                         gp_y_train_pred_sorted.detach().cpu().numpy().flatten(),
                                         yerr=gp_y_train_pred_std_sorted.detach().cpu().numpy().flatten(),
                                         fmt='*', alpha=0.05, label='y pred', color='C0', ecolor='C0')
                            plt.scatter(np.arange(len(indices_train_pred)),
                                        gp_y_train_pred_sorted.detach().cpu().numpy(),
                                        marker='*', alpha=0.2, s=10, color='C0')
                            plt.legend()
                            plt.title('y predictions and uncertainty on train set')
                            plt.savefig(os.path.join(save_folder, 'gp_y_train_uncertainty.pdf'))
                            plt.close()

                            y_test_sorted, indices_test_pred = torch.sort(-y_val, dim=0)
                            gp_y_test_pred_sorted = pred_y_test_mean[indices_test_pred].view(-1, 1)
                            gp_y_test_pred_std_sorted = pred_y_test_std[indices_test_pred].view(-1, 1)
                            plt.scatter(np.arange(len(indices_test_pred)), y_test_sorted.cpu().numpy(),
                                        label='y true',
                                        marker='+', color='C1', s=15)
                            plt.errorbar(np.arange(len(indices_test_pred)),
                                         gp_y_test_pred_sorted.detach().cpu().numpy().flatten(),
                                         yerr=gp_y_test_pred_std_sorted.detach().cpu().numpy().flatten(),
                                         marker='*', alpha=0.05, label='erry pred', color='C0', ecolor='C0')
                            plt.scatter(np.arange(len(indices_test_pred)),
                                        gp_y_test_pred_sorted.detach().cpu().numpy().flatten(),
                                        marker='*', color='C0', alpha=0.2, s=10)
                            plt.legend()
                            plt.title('y predictions and uncertainty on test set')
                            plt.savefig(os.path.join(save_folder, 'gp_y_test_uncertainty.pdf'))
                            plt.close()

                            if x_train.size(1) == 2:
                                from itertools import product

                                fig = plt.figure()
                                ax = fig.gca(projection='3d')
                                ax.scatter(x_train[:, 0].cpu().numpy(), x_train[:, 1].cpu().numpy(),
                                           -y_train.cpu().numpy(), label=f"y train true", marker="+",
                                           alpha=0.05, color='C0')
                                ax.scatter(x_val[:, 0].cpu().numpy(), x_val[:, 1].cpu().numpy(),
                                           -y_val.cpu().numpy(), label=f"y test true", marker="+",
                                           alpha=0.05, color='C4')
                                X, Y = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
                                X_all = np.array(list(product(X, Y)))
                                X, Y = np.meshgrid(X, Y)
                                pred_y_train_mean_grid = gp_obj_model.posterior(
                                    torch.from_numpy(X_all).to(**tkwargs)).mean
                                pred_y_train_std_grid = gp_obj_model.posterior(
                                    torch.from_numpy(X_all).to(**tkwargs)).variance.sqrt()
                                Z = pred_y_train_mean_grid.view(100, 100).detach().cpu().numpy()
                                ZZ = pred_y_train_std_grid.view(100, 100).detach().cpu().numpy()
                                ax.plot_surface(X, Y, Z, label=f"y_train pred mean", alpha=0.15, color='C1')
                                # ax.plot_surface(X, Y, Z + ZZ, label=f"y_train pred mean", alpha=0.15, color='C2')
                                # ax.plot_surface(X, Y, Z - ZZ, label=f"y_train pred mean", alpha=0.15, color='C3')
                                plt.title('GP pred')
                                plt.savefig(os.path.join(save_folder, f"gp_pred_viz_3D{name}.pdf"))
                                plt.close()

                            # reset x_train and x_test
                            if subset:
                                x_train = torch.load(os.path.join(data_folder, f"train_x.pt")).to(**tkwargs)
                                # x_test = torch.load(os.path.join(gp_run_folder, f"test_x.pt")).to(**tkwargs)

                        gp_obj_model_name.pop()
                        gp_obj_model_name.pop()
                    gp_obj_model_name.pop()
                gp_obj_model_name.pop()
            gp_obj_model_name.pop()
            gp_obj_model_name.pop()
        gp_obj_model_name.pop()
