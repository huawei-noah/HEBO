# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)

import GPy
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, FloatTensor, LongTensor

from ..base_model import BaseModel
from ..layers import EmbTransform, OneHotTransform
from ..scalers import TorchMinMaxScaler, TorchStandardScaler
from ..util import filter_nan


class GPyMLPGP(BaseModel):
    """
    Input warped GP model implemented using GPy instead of GPyTorch

    Why doing so:
    - Input warped GP
    """
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        if num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])
        self.xscaler    = TorchMinMaxScaler((-1, 1))
        self.yscaler    = TorchStandardScaler()
        self.verbose    = self.conf.get('verbose', False)
        self.num_epochs = self.conf.get('num_epochs', 200)
        self.num_restarts = self.conf.get('num_restarts', 10)

    def fit_scaler(self, Xc : FloatTensor, y : FloatTensor):
        if Xc is not None and Xc.shape[1] > 0:
            self.xscaler.fit(Xc)
        self.yscaler.fit(y)

    def trans(self, Xc : Tensor, Xe : Tensor, y : Tensor = None):
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = torch.zeros(Xe.shape[0], 0)

        if Xe is None or Xe.shape[1] == 0:
            Xe_t = torch.zeros(Xc.shape[0], 0)
        else:
            Xe_t = self.one_hot(Xe.long())
        Xall = torch.cat([Xc_t, Xe_t], dim = 1)

        if y is not None:
            y_t = self.yscaler.transform(y)
            return Xall.numpy(), y_t.numpy()
        return Xall.numpy()

    def fit(self, Xc : FloatTensor, Xe : LongTensor, y : LongTensor): 
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'all')
        self.fit_scaler(Xc, y)
        X, y = self.trans(Xc, Xe, y)

        kern    = GPy.kern.src.mlp.MLP(input_dim = X.shape[1], ARD = True)
        self.gp = GPy.models.GPRegression(X, y, kern)
        self.gp.kern.variance    = np.var(y)
        self.gp.kern.lengthscale = np.std(X, axis = 0).clip(min = 0.02)
        self.gp.likelihood.variance = 1e-2 * np.var(y)

        self.gp.kern.variance.set_prior(GPy.priors.Gamma(0.5, 0.5), warning = False)
        self.gp.likelihood.variance.set_prior(GPy.priors.LogGaussian(-4.63, 0.5), warning = False)

        self.gp.optimize_restarts(max_iters = self.num_epochs, verbose = self.verbose, num_restarts = self.num_restarts, robust = True)
        if self.verbose:
            print(self.gp.likelihood.variance, flush = True)
            print(self.gp.likelihood.variance[0], flush = True)
        return self

    def predict(self, Xc : FloatTensor, Xe : LongTensor) -> (FloatTensor, FloatTensor):
        Xall    = self.trans(Xc, Xe)
        py, ps2 = self.gp.predict(Xall)
        mu      = self.yscaler.inverse_transform(FloatTensor(py).view(-1, 1))
        var     = (self.yscaler.std**2 * FloatTensor(ps2).view(-1, 1))
        return mu, var.clamp(torch.finfo(var.dtype).eps)

    def sample_f(self):
        raise NotImplementedError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def noise(self):
        var_normalized = self.gp.likelihood.variance[0]
        return (var_normalized * self.yscaler.std**2).view(self.num_out)
