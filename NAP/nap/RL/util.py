# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from multiprocessing import Pool

import GPy
import numpy as np
import os
import pickle as pkl
import torch

import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from nap.RL.utils_gp import MixtureKernel

def compute_gp_param(X, Y):
    D = X.shape[1]
    kernel_variance = 1.0
    kernel_lengthscale = 1.0

    kernels = []

    kernels.append(GPy.kern.RBF(input_dim=D,
                                variance=kernel_variance,
                                lengthscale=kernel_lengthscale,
                                ARD=True))

    normalizer = False
    for i in range(len(kernels)):
        gp = GPy.models.sparse_gp_regression.SparseGPRegression(X, Y[:, None],
                                                        kernel=kernels[i],
                                                        normalizer=normalizer,
                                                        num_inducing=30)
        gp.optimize()
        return np.array(gp.kern.variance), np.array(gp.kern.lengthscale), np.array(gp.Gaussian_noise.variance)


def fit_gp_gpytorch(X, Y, save_dir, cat_dims=None, trainXmean=None, trainXstd=None):
    if cat_dims is not None:
        X, Y = torch.from_numpy(X).to('cuda:0'), torch.from_numpy(Y).to('cuda:0')
        stdY = (Y - Y.mean()) / Y.std()
        cat_dims = [i+X.shape[-1] for i in cat_dims if i < 0]
        cont_dims = [i for i in range(X.shape[-1]) if i not in cat_dims]
        normX = X.clone()
        normX[:, cont_dims] = ((normX[:, cont_dims] - normX[:, cont_dims].min(0)[0].view(1, -1))
                 / (normX[:, cont_dims].max(0)[0].view(1, -1) - normX[:, cont_dims].min(0)[0].view(1, -1)))
        model = SingleTaskGP(
            train_X=normX,
            train_Y=stdY.view(-1, 1),
            covar_module=MixtureKernel(
                categorical_dims=cat_dims,
                continuous_dims=cont_dims
            ))
        try:
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            mll = fit_gpytorch_mll(mll=mll)
        except botorch.exceptions.errors.ModelFittingError as e:
            print(e)
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            mll = fit_gpytorch_mll_torch(mll=mll)

        print('Save model at', save_dir)
        with torch.no_grad():
            print('cat lengthscale:', model.covar_module.categorical_kern.lengthscale)
            print('num lengthscale:', model.covar_module.continuous_kern.lengthscale)
            print('lengthscale:', model.covar_module.lengthscale)
            print('noise:      ', model.likelihood.noise)
            torch.save(model, save_dir)

    else:
        X, Y = torch.from_numpy(X).to('cuda:0'), torch.from_numpy(Y).to('cuda:0')
        stdX = (X - torch.from_numpy(trainXmean).to(X)) / torch.from_numpy(trainXstd).to(X)
        stdY = (Y - Y.mean()) / Y.std()
        model = SingleTaskGP(stdX, stdY.view(-1, 1))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll = fit_gpytorch_mll(mll=mll)

        print('Save model at', save_dir)
        with torch.no_grad():
            torch.save(model, save_dir)



def compute_cond_gps(dataset_paths, saved_models_dir, **kwargs):
    for path in dataset_paths:
        data = pkl.load(open(path, "rb"))
        model_name = path.split("/")[-1].split(".")[0] + "_gp_model.pt"
        model_path = os.path.join(saved_models_dir, model_name)
        fit_gp_gpytorch(X=data['domain'], Y=data['accs'], save_dir=model_path, **kwargs)


def print_best_gp_params(X, Y):

    returns = []
    with Pool(processes=X.shape[0]) as pool:
        gp_procs = [pool.apply_async(compute_gp_param, (X[p], Y[p])) for p in range(X.shape[0]) for _ in range(3)]

        for proc in gp_procs:
            returns.append(proc.get())

    mean_kernel_variance, mean_kernel_lengthscale, mean_noise_variance = zip(*returns)

    np.set_printoptions(suppress=True)
    print("kernel_variance", np.array(mean_kernel_variance).mean(0))
    print("kernel_lengthscale", np.array(mean_kernel_lengthscale).mean(0))
    print("noise_variance", np.array(mean_noise_variance).mean(0))

    breakpoint()
