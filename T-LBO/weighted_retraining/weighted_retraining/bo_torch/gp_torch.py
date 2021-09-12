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
import time
from argparse import ArgumentParser
from typing import Dict, Any, Optional, List, Iterable

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import acquisition
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.models.transforms import Warp
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from botorch.optim.utils import _filter_kwargs
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, MaternKernel, Kernel
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior, Prior, LogNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import power_transform
from torch import Tensor

from utils.utils_cmd import parse_dict
from weighted_retraining.weighted_retraining.bo_torch.fit import fit_gpytorch_torch
from weighted_retraining.weighted_retraining.bo_torch.optimize import optimize_acqf_torch


class CustomWarp(Warp):
    r"""A transform that uses learned input warping functions.

    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.

    Add Interval constraints on concentration coefficients instead of only lower bounds
    """

    _max_concentration_level = 10

    def __init__(
            self,
            indices: List[int],
            transform_on_train: bool = True,
            transform_on_eval: bool = True,
            transform_on_preprocess: bool = False,
            reverse: bool = False,
            eps: float = 1e-7,
            concentration1_prior: Optional[Prior] = None,
            concentration0_prior: Optional[Prior] = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when preprocessing. Default: False.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            concentration1_prior: A prior distribution on the concentration1 parameter
                of the Kumaraswamy distribution.
            concentration0_prior: A prior distribution on the concentration0 parameter
                of the Kumaraswamy distribution.


        """
        super().__init__(indices=indices,
                         transform_on_train=transform_on_train,
                         transform_on_eval=transform_on_eval,
                         transform_on_preprocess=transform_on_preprocess,
                         reverse=reverse,
                         eps=eps,
                         concentration1_prior=concentration1_prior,
                         concentration0_prior=concentration0_prior

                         )

        for i in (0, 1):
            p_name = f"concentration{i}"
            constraint = Interval(
                self._min_concentration_level,
                self._max_concentration_level,
                transform=None,
                initial_value=torch.tensor(1.)
            )
            self.register_constraint(param_name=p_name, constraint=constraint)


class SVGP(ApproximateGP):
    def __init__(self, inducing_points: Tensor, covar_module: Optional[gpytorch.kernels.Kernel] = None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, x: Tensor):
        self.eval()
        return self.forward(x)


def query_covar(covar_name: str, scale: bool, outputscale: float, lscales: Tensor, **kwargs) -> Kernel:
    lengthscale_prior = GammaPrior(3.0, 6.0)
    kws = dict(lengthscale_prior=lengthscale_prior, ard_num_dims=lscales.shape[-1], )
    if covar_name.lower()[:6] == 'matern':
        kernel_class = MaternKernel
        if covar_name[-2:] == '52':
            kws['nu'] = 2.5
        elif covar_name[-2:] == '32':
            kws['nu'] = 1.5
        elif covar_name[-2:] == '12':
            kws['nu'] = .5
        else:
            raise ValueError(covar_name)
    elif covar_name.lower() == 'rbf':
        kernel_class = RBFKernel
    else:
        raise ValueError(covar_name)
    kws.update(**kwargs)

    kernel = kernel_class(**kws)
    kernel.lengthscale = lscales
    if scale:
        kernel = ScaleKernel(kernel, outputscale_prior=GammaPrior(2.0, 0.15))
        kernel.outputscale = outputscale

    return kernel


def gp_torch_train(train_x: Tensor, train_y: Tensor, n_inducing_points: int, tkwargs: Dict[str, Any],
                   init, scale: bool, covar_name: str, gp_file: Optional[str], save_file: str, input_wp: bool,
                   outcome_transform: Optional[OutcomeTransform] = None,
                   options: Dict[str, Any] = None) -> SingleTaskGP:
    assert train_y.ndim > 1, train_y.shape
    assert gp_file or init, (gp_file, init)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if init:
        # build hyp
        print("Initialize GP hparams...")
        print("Doing Kmeans init...")
        assert n_inducing_points > 0, n_inducing_points
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing_points,
            batch_size=min(10000, train_x.shape[0]),
            n_init=25
        )
        start_time = time.time()
        kmeans.fit(train_x.cpu().numpy())
        end_time = time.time()
        print(f"K means took {end_time - start_time:.1f}s to finish...")
        inducing_points = torch.from_numpy(kmeans.cluster_centers_.copy())

        output_scale = None
        if scale:
            output_scale = train_y.var().item()
        lscales = torch.empty(1, train_x.shape[1])
        for i in range(train_x.shape[1]):
            lscales[0, i] = torch.pdist(train_x[:, i].view(-1, 1)).median().clamp(min=0.01)
        base_covar_module = query_covar(covar_name=covar_name, scale=scale, outputscale=output_scale, lscales=lscales)

        covar_module = InducingPointKernel(base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

        input_warp_tf = None
        if input_wp:
            # Apply input warping
            # initialize input_warping transformation
            input_warp_tf = CustomWarp(
                indices=list(range(train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
            )

        model = SingleTaskGP(train_x, train_y,
                             covar_module=covar_module, likelihood=likelihood,
                             input_transform=input_warp_tf, outcome_transform=outcome_transform)
    else:
        # load model
        output_scale = 1  # will be overwritten when loading model
        lscales = torch.ones(train_x.shape[1])  # will be overwritten when loading model
        base_covar_module = query_covar(covar_name=covar_name, scale=scale, outputscale=output_scale, lscales=lscales)
        covar_module = InducingPointKernel(base_covar_module,
                                           inducing_points=torch.empty(n_inducing_points, train_x.shape[1]),
                                           likelihood=likelihood)

        input_warp_tf = None
        if input_wp:
            # Apply input warping
            # initialize input_warping transformation
            input_warp_tf = Warp(
                indices=list(range(train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
            )
        model = SingleTaskGP(train_x, train_y, covar_module=covar_module, likelihood=likelihood,
                             input_transform=input_warp_tf, outcome_transform=outcome_transform)
        print("Loading GP from file")
        state_dict = torch.load(gp_file)
        model.load_state_dict(state_dict)

    print("GP regression")
    start_time = time.time()
    model.to(**tkwargs)
    model.train()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # set approx_mll to False since we are using an exact marginal log likelihood
    # fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, approx_mll=False, options=options)
    fit_gpytorch_torch(
        mll,
        options=options,
        approx_mll=False,
        clip_by_value=True if input_wp else False,
        clip_value=10.0
    )
    end_time = time.time()
    print(f"Regression took {end_time - start_time:.1f}s to finish...")

    print("Save GP model...")
    torch.save(model.state_dict(), save_file)
    print("Done training of GP.")

    model.eval()
    return model


def gp_fit_test(x_train: Tensor, y_train: Tensor, error_train: Tensor,
                x_test: Tensor, y_test: Tensor, error_test: Tensor,
                gp_obj_model: SingleTaskGP, gp_error_model: SingleTaskGP,
                tkwargs: Dict[str, Any], gp_test_folder: str,
                obj_out_wp: bool = False, err_out_wp: bool = False) -> None:
    """
    1) Estimates mean test error between predicted and the true objective function values.
    2) Estimates mean test error between predicted recon. error by the gp_model and the true recon. error of the vae_model.
    :param x_train: normalised points at which the gps were trained
    :param y_train: objective value function corresponding to x_train that were used as targets of `gp_obj_model`
    :param error_train: reconstruction error value at points x_train that were used as targets of `gp_error_model`
    :param x_test: normalised test points
    :param y_test: objective value function corresponding to x_test
    :param error_test: reconstruction error at test points
    :param gp_obj_model: the gp model trained to predict the black box objective function values
    :param gp_error_model: the gp model trained to predict reconstruction error
    :param tkwargs: dict of type and device
    :param gp_test_folder: folder to save test results
    :param obj_out_wp: if the `gp_obj_model` was trained with output warping then need to apply the same transform
    :param err_out_wp: if the `gp_error_model` was trained with output warping then need to apply the same transform
    :return: (Sum_i||true_y_i - pred_y_i||^2 / n_points, Sum_i||true_recon_i - pred_recon_i||^2 / n_points)
    """
    do_robust = True if gp_error_model is not None else False
    if not os.path.exists(gp_test_folder):
        os.mkdir(gp_test_folder)

    gp_obj_model.eval()
    gp_obj_model.to(tkwargs['device'])
    y_train = y_train.view(-1)
    if do_robust:
        gp_error_model.eval()
        gp_error_model.to(tkwargs['device'])
        error_train = error_train.view(-1)

    with torch.no_grad():
        if obj_out_wp:
            Y_numpy = y_train.cpu().numpy()
            if Y_numpy.min() <= 0:
                y_train = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='yeo-johnson'))
            else:
                y_train = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='box-cox'))
                if y_train.std() < 0.5:
                    Y_numpy = y_train.numpy()
                    y_train = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='yeo-johnson')).to(
                        x_train)

            Y_numpy = y_test.cpu().numpy()
            if Y_numpy.min() <= 0:
                y_test = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='yeo-johnson'))
            else:
                y_test = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='box-cox'))
                if y_test.std() < 0.5:
                    Y_numpy = y_test.numpy()
                    y_test = torch.FloatTensor(power_transform(Y_numpy / Y_numpy.std(), method='yeo-johnson')).to(
                        x_test)

        y_train = y_train.view(-1).to(**tkwargs)
        y_test = y_test.view(-1).to(**tkwargs)

        gp_obj_val_model_mse_train = (gp_obj_model.posterior(x_train).mean.view(-1) - y_train).pow(2).div(len(y_train))
        gp_obj_val_model_mse_test = (gp_obj_model.posterior(x_test).mean.view(-1) - y_test).pow(2).div(len(y_test))
        torch.save(gp_obj_val_model_mse_train, os.path.join(gp_test_folder, 'gp_obj_val_model_mse_train.npz'))
        torch.save(gp_obj_val_model_mse_test, os.path.join(gp_test_folder, 'gp_obj_val_model_test.npz'))
        print(f'GP training fit on objective value: MSE={gp_obj_val_model_mse_train.sum().item():.5f}')
        print(f'GP testing fit on objective value: MSE={gp_obj_val_model_mse_test.sum().item():.5f}')

        if do_robust:
            if err_out_wp:
                error_train = error_train.view(-1, 1)
                R_numpy = error_train.cpu().numpy()
                if R_numpy.min() <= 0:
                    error_train = torch.FloatTensor(power_transform(R_numpy / R_numpy.std(), method='yeo-johnson'))
                else:
                    error_train = torch.FloatTensor(power_transform(R_numpy / R_numpy.std(), method='box-cox'))
                    if error_train.std() < 0.5:
                        R_numpy = error_train.numpy()
                        error_train = torch.FloatTensor(
                            power_transform(R_numpy / R_numpy.std(), method='yeo-johnson')).to(x_train)

                R_numpy = error_test.cpu().numpy()
                if R_numpy.min() <= 0:
                    error_test = torch.FloatTensor(power_transform(R_numpy / R_numpy.std(), method='yeo-johnson'))
                else:
                    error_test = torch.FloatTensor(power_transform(R_numpy / R_numpy.std(), method='box-cox'))
                    if error_test.std() < 0.5:
                        R_numpy = error_test.numpy()
                        error_test = torch.FloatTensor(
                            power_transform(R_numpy / R_numpy.std(), method='yeo-johnson')).to(x_test)

            error_train = error_train.view(-1).to(**tkwargs)
            error_test = error_test.view(-1).to(**tkwargs)

            pred_recon_train = gp_error_model.posterior(x_train).mean.view(-1)
            pred_recon_test = gp_error_model.posterior(x_test).mean.view(-1)

            gp_error_model_mse_train = (error_train - pred_recon_train).pow(2).div(len(error_train))
            gp_error_model_mse_test = (error_test - pred_recon_test).pow(2).div(len(error_test))
            torch.save(gp_error_model_mse_train, os.path.join(gp_test_folder, 'gp_error_model_mse_train.npz'))
            torch.save(gp_error_model_mse_test, os.path.join(gp_test_folder, 'gp_error_model_mse_test.npz'))
            print(f'GP training fit on reconstruction errors: MSE={gp_error_model_mse_train.sum().item():.5f}')
            print(f'GP testing fit on reconstruction errors: MSE={gp_error_model_mse_test.sum().item():.5f}')
            torch.save(error_test, os.path.join(gp_test_folder, f"true_rec_err_z.pt"))
            torch.save(error_train, os.path.join(gp_test_folder, f"error_train.pt"))

        torch.save(x_train, os.path.join(gp_test_folder, f"train_x.pt"))
        torch.save(x_test, os.path.join(gp_test_folder, f"test_x.pt"))
        torch.save(y_train, os.path.join(gp_test_folder, f"y_train.pt"))
        torch.save(x_test, os.path.join(gp_test_folder, f"X_test.pt"))
        torch.save(y_test, os.path.join(gp_test_folder, f"y_test.pt"))

        # y plots
        plt.hist(y_train.cpu().numpy(), bins=100, label='y train', alpha=0.5, density=True)
        plt.hist(gp_obj_model.posterior(x_train).mean.view(-1).detach().cpu().numpy(), bins=100,
                 label='y pred', alpha=0.5, density=True)
        plt.legend()
        plt.title('Training set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_train.pdf'))
        plt.close()

        plt.hist(gp_obj_val_model_mse_train.detach().cpu().numpy(), bins=100, alpha=0.5, density=True)
        plt.title('MSE of gp_obj_val model on training set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_train_mse.pdf'))
        plt.close()

        plt.hist(y_test.cpu().numpy(), bins=100, label='y true', alpha=0.5, density=True)
        plt.hist(gp_obj_model.posterior(x_test).mean.detach().cpu().numpy(), bins=100,
                 alpha=0.5, label='y pred', density=True)
        plt.legend()
        plt.title('Validation set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_test.pdf'))
        plt.close()

        plt.hist(gp_obj_val_model_mse_test.detach().cpu().numpy(), bins=100, alpha=0.5, density=True)
        plt.title('MSE of gp_obj_val model on validation set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_test_mse.pdf'))
        plt.close()

        if do_robust:
            # error plots
            plt.hist(error_train.cpu().numpy(), bins=100, label='error train', alpha=0.5, density=True)
            plt.hist(gp_error_model.posterior(x_train).mean.detach().cpu().numpy(), bins=100,
                     label='error pred', alpha=0.5, density=True)
            plt.legend()
            plt.title('Training set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_train.pdf'))
            plt.close()

            plt.hist(gp_error_model_mse_train.detach().cpu().numpy(), bins=100, alpha=0.5, density=True)
            plt.title('MSE of gp_error model on training set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_train_mse.pdf'))
            plt.close()

            plt.hist(error_test.cpu().numpy(), bins=100, label='error true', alpha=0.5, density=True)
            plt.hist(gp_error_model.posterior(x_test).mean.detach().cpu().numpy(), bins=100,
                     alpha=0.5, label='error pred', density=True)
            plt.legend()
            plt.title('Validation set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_test.pdf'))
            plt.close()

            plt.hist(gp_error_model_mse_test.detach().cpu().numpy(), bins=100, alpha=0.5, density=True)
            plt.title('MSE of gp_error model on validation set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_test_mse.pdf'))
            plt.close()

            # y-error plots
            y_train_sorted, indices_train = torch.sort(y_train)
            error_train_sorted = error_train[indices_train]
            gp_y_train_pred_sorted, indices_train_pred = torch.sort(gp_obj_model.posterior(x_train).mean.view(-1))
            gp_r_train_pred_sorted = (gp_error_model.posterior(x_train).mean.view(-1))[indices_train_pred]
            plt.scatter(y_train_sorted.cpu().numpy(), error_train_sorted.cpu().numpy(), label='true', marker='+')
            plt.scatter(gp_y_train_pred_sorted.detach().cpu().numpy(), gp_r_train_pred_sorted.detach().cpu().numpy(),
                        label='pred', marker='*')
            plt.xlabel('y train targets')
            plt.ylabel('recon. error train targets')
            plt.title('y_train vs. error_train')
            plt.legend()
            plt.savefig(os.path.join(gp_test_folder, 'scatter_obj_error_train.pdf'))
            plt.close()

            y_test_std_sorted, indices_test = torch.sort(y_test)
            error_test_sorted = error_test[indices_test]
            gp_y_test_pred_sorted, indices_test_pred = torch.sort(gp_obj_model.posterior(x_test).mean.view(-1))
            gp_r_test_pred_sorted = (gp_error_model.posterior(x_test).mean.view(-1))[indices_test_pred]
            plt.scatter(y_test_std_sorted.cpu().numpy(), error_test_sorted.cpu().numpy(), label='true', marker='+')
            plt.scatter(gp_y_test_pred_sorted.detach().cpu().numpy(), gp_r_test_pred_sorted.detach().cpu().numpy(),
                        label='pred', marker='*')
            plt.xlabel('y test targets')
            plt.ylabel('recon. error test targets')
            plt.title('y_test vs. error_test')
            plt.legend()
            plt.savefig(os.path.join(gp_test_folder, 'scatter_obj_error_test.pdf'))
            plt.close()

            # error var plots
            error_train_sorted, indices_train_pred = torch.sort(error_train)
            # error_train_sorted = error_train
            # indices_train_pred = np.arange(len(error_train))
            gp_r_train_pred_sorted = gp_error_model.posterior(x_train).mean[indices_train_pred].view(-1)
            gp_r_train_pred_std_sorted = gp_error_model.posterior(x_train).variance.view(-1).sqrt()[indices_train_pred]
            plt.scatter(np.arange(len(indices_train_pred)), error_train_sorted.cpu().numpy(),
                        label='err true', marker='+', color='C1', s=15)
            plt.errorbar(np.arange(len(indices_train_pred)),
                         gp_r_train_pred_sorted.detach().cpu().numpy().flatten(),
                         yerr=gp_r_train_pred_std_sorted.detach().cpu().numpy().flatten(),
                         fmt='*', alpha=0.05, label='err pred', color='C0', ecolor='C0')
            plt.scatter(np.arange(len(indices_train_pred)), gp_r_train_pred_sorted.detach().cpu().numpy(),
                        marker='*', alpha=0.2, s=10, color='C0')
            # plt.scatter(np.arange(len(indices_train_pred)),
            #             (gp_r_train_pred_sorted + gp_r_train_pred_std_sorted).detach().cpu().numpy(),
            #             label='err pred mean+std', marker='.')
            # plt.scatter(np.arange(len(indices_train_pred)),
            #             (gp_r_train_pred_sorted - gp_r_train_pred_std_sorted).detach().cpu().numpy(),
            #             label='err pred mean-std', marker='.')
            plt.legend()
            plt.title('error predictions and uncertainty on train set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_train_uncertainty.pdf'))
            plt.close()

            error_test_sorted, indices_test_pred = torch.sort(error_test)
            # error_test_sorted = error_test
            # indices_test_pred = np.arange(len(error_test_sorted))
            gp_r_test_pred_sorted = gp_error_model.posterior(x_test).mean.view(-1)[indices_test_pred]
            gp_r_test_pred_std_sorted = gp_error_model.posterior(x_test).variance.view(-1).sqrt()[indices_test_pred]
            plt.scatter(np.arange(len(indices_test_pred)), error_test_sorted.cpu().numpy(), label='err true',
                        marker='+', color='C1', s=15)
            plt.errorbar(np.arange(len(indices_test_pred)),
                         gp_r_test_pred_sorted.detach().cpu().numpy().flatten(),
                         yerr=gp_r_test_pred_std_sorted.detach().cpu().numpy().flatten(),
                         marker='*', alpha=0.05, label='err pred', color='C0', ecolor='C0')
            plt.scatter(np.arange(len(indices_test_pred)), gp_r_test_pred_sorted.detach().cpu().numpy().flatten(),
                        marker='*', color='C0', alpha=0.2, s=10)
            # plt.scatter(np.arange(len(indices_test_pred)),
            #             (gp_r_test_pred_sorted + gp_r_test_pred_std_sorted).detach().cpu().numpy(),
            #             label='err pred mean+std', marker='.')
            # plt.scatter(np.arange(len(indices_test_pred)),
            #             (gp_r_test_pred_sorted - gp_r_test_pred_std_sorted).detach().cpu().numpy(),
            #             label='err pred mean-std', marker='.')
            plt.legend()
            plt.title('error predictions and uncertainty on test set')
            plt.savefig(os.path.join(gp_test_folder, 'gp_error_test_uncertainty.pdf'))
            plt.close()

        # y var plots
        y_train_std_sorted, indices_train = torch.sort(y_train)
        gp_y_train_pred_sorted = gp_obj_model.posterior(x_train).mean[indices_train].view(-1)
        gp_y_train_pred_std_sorted = gp_obj_model.posterior(x_train).variance.sqrt()[indices_train].view(-1)
        plt.scatter(np.arange(len(indices_train)), y_train_std_sorted.cpu().numpy(),
                    label='y true', marker='+', color='C1', s=15)
        plt.scatter(np.arange(len(indices_train)), gp_y_train_pred_sorted.detach().cpu().numpy(),
                    marker='*', alpha=0.2, s=10, color='C0')
        plt.errorbar(np.arange(len(indices_train)),
                     gp_y_train_pred_sorted.detach().cpu().numpy().flatten(),
                     yerr=gp_y_train_pred_std_sorted.detach().cpu().numpy().flatten(),
                     fmt='*', alpha=0.05, label='y pred', color='C0', ecolor='C0')
        # plt.scatter(np.arange(len(indices_train_pred)),
        #             (gp_y_train_pred_sorted+gp_y_train_pred_std_sorted).detach().cpu().numpy(),
        #             label='y pred mean+std', marker='.')
        # plt.scatter(np.arange(len(indices_train_pred)),
        #             (gp_y_train_pred_sorted-gp_y_train_pred_std_sorted).detach().cpu().numpy(),
        #             label='y pred mean-std', marker='.')
        plt.legend()
        plt.title('y predictions and uncertainty on train set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_val_train_uncertainty.pdf'))
        plt.close()

        y_test_std_sorted, indices_test = torch.sort(y_test)
        gp_y_test_pred_sorted = gp_obj_model.posterior(x_test).mean.view(-1)[indices_test]
        gp_y_test_pred_std_sorted = gp_obj_model.posterior(x_test).variance.view(-1).sqrt()[indices_test]
        plt.scatter(np.arange(len(indices_test)), y_test_std_sorted.cpu().numpy(),
                    label='y true', marker='+', color='C1', s=15)
        plt.errorbar(np.arange(len(indices_test)),
                     gp_y_test_pred_sorted.detach().cpu().numpy().flatten(),
                     yerr=gp_y_test_pred_std_sorted.detach().cpu().numpy().flatten(),
                     fmt='*', alpha=0.05, label='y pred', color='C0', ecolor='C0')
        plt.scatter(np.arange(len(indices_test)), gp_y_test_pred_sorted.detach().cpu().numpy(),
                    marker='*', alpha=0.2, s=10, color='C0')
        # plt.scatter(np.arange(len(indices_test_pred)),
        #             (gp_y_test_pred_sorted + gp_y_test_pred_std_sorted).detach().cpu().numpy(),
        #             label='y pred mean+std', marker='.')
        # plt.scatter(np.arange(len(indices_test_pred)),
        #             (gp_y_test_pred_sorted - gp_y_test_pred_std_sorted).detach().cpu().numpy(),
        #             label='y pred mean-std', marker='.')
        plt.legend()
        plt.title('y predictions and uncertainty on test set')
        plt.savefig(os.path.join(gp_test_folder, 'gp_obj_val_test_uncertainty.pdf'))
        plt.close()


def query_acq_func(acq_func_id: str, acq_func_kwargs: Dict[str, Any], gp_model: SingleTaskGP, q: int,
                   num_MC_samples_acq: int):
    if not hasattr(AnalyticAcquisitionFunction, acq_func_id):
        # use MC version of acq function
        acq_func_id = f'q{acq_func_id}'
        resampler = SobolQMCNormalSampler(num_samples=num_MC_samples_acq, resample=True).to(gp_model.train_inputs[0])
        acq_func_kwargs['sampler'] = resampler

    acq_func_class = getattr(acquisition, acq_func_id)
    acq_func = acq_func_class(gp_model, **_filter_kwargs(acq_func_class, **acq_func_kwargs))
    return acq_func


def bo_loop(gp_model: SingleTaskGP, acq_func_id: str,
            acq_func_kwargs: Dict[str, Any], acq_func_opt_kwargs: Dict[str, Any],
            bounds: Tensor, tkwargs: Dict[str, Any], q: int,
            num_restarts: int, raw_initial_samples, seed: int, num_MC_sample_acq: int) -> Iterable[Any]:
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)

    # put on proper device

    # we want to maximize
    fmax = torch.quantile(gp_model.train_targets, .9).item()
    print(f"Using good point cutoff {fmax:.2f}")

    device = gp_model.train_inputs[0].device

    bounds = bounds.to(**tkwargs)
    gp_model.eval()

    acq_func_kwargs['best_f'] = fmax
    acq_func = query_acq_func(acq_func_id=acq_func_id, acq_func_kwargs=acq_func_kwargs, gp_model=gp_model,
                              q=q, num_MC_samples_acq=num_MC_sample_acq)  # if q is 1 use analytic acquisitions
    acq_func.to(**tkwargs)

    options = {'batch_limit': 100} if acq_func_opt_kwargs == {} else acq_func_opt_kwargs
    print("Start acquisition function optimization...")
    if q == 1:
        # use optimize_acq (with LBFGS)
        candidate, acq_value = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=num_restarts,
                                             raw_samples=raw_initial_samples, return_best_only=True, options=options
                                             )
    else:
        candidate, acq_value = optimize_acqf_torch(acq_function=acq_func, bounds=bounds, q=q,
                                                   num_restarts=num_restarts, raw_samples=raw_initial_samples,
                                                   return_best_only=True, options=options,
                                                   )
    print(f"Acquired {candidate} with acquisition value {acq_value}")
    return candidate.to(device=device)


def add_gp_torch_args(parser: ArgumentParser):
    parser.register('type', dict, parse_dict)
    gp_sparse_group = parser.add_argument_group("Sparse GP")
    gp_sparse_group.add_argument("--n_inducing_points", type=int, default=500)
    gp_sparse_group.add_argument("--n_rand_points", type=int, default=8000)
    gp_sparse_group.add_argument("--n_best_points", type=int, default=2000)
    gp_sparse_group.add_argument("--invalid_score", type=float, default=-4.0)

    gp_group = parser.add_argument_group("GP tuning")
    gp_group.add_argument("--scale", type=int, choices=(0, 1), default=1, help="Whether to use a scaled covar module")
    gp_group.add_argument("--covar-name", type=str, default='matern-5/2',
                          help="Name of the kernel to use (`matern-5/2`, `RBF`...)")

    gp_group = parser.add_argument_group("Acquisition function maximisation")
    gp_group.add_argument("--acq-func-id", type=str, default='ExpectedImprovement',
                          help="Name of the acquisition function to use (`ExpectedImprovement`, `UpperConfidenceBound`...)")
    gp_group.add_argument("--acq-func-kwargs", type=dict, default={}, help="Acquisition function kwargs")
    gp_group.add_argument("--acq-func-opt-kwargs", type=dict, default={},
                          help="Acquisition function Optimisation kwargs")
    gp_group.add_argument("--q", type=int, default=1, help="Acquisition batch size")
    gp_group.add_argument("--num-restarts", type=int, default=100, help="Number of start points")
    gp_group.add_argument("--raw-initial-samples", type=int, default=1000,
                          help="Number of initial points used to find start points")
    gp_group.add_argument("--num-MC-sample-acq", type=int, default=256,
                          help="Number of samples to use to evaluate posterior distribution")

    return parser
