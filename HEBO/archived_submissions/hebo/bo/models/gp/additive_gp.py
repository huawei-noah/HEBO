# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import pickle

from torch import Tensor
from pathlib import Path
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..base_model import BaseModel
from ..layers import EmbTransform

class GPyTorchModel(gpytorch.models.ExactGP):
    def __init__(self, 
            x   : torch.Tensor,
            xe  : torch.Tensor,
            y   : torch.Tensor,
            lik : GaussianLikelihood, 
            **conf):
        super().__init__((x, xe), y.squeeze(), lik)
        self.grps = conf.get('additive_grps', None)

    def forward(self, x, xe):
        pass

class AddGP(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        assert self.num_out == 1, 'GP only supports single-output'
        self.lr           = conf.get('lr', 3e-2)
        self.num_epochs   = conf.get('num_epochs', 500)
        self.verbose      = conf.get('verbose', False)
        self.print_every  = conf.get('print_every', 10)
        self.noise_free   = conf.get('noise_free', False)
        self.additive_grp = conf.get('additive_grp', None) # None for fully coupled model

    def fit(self, Xc : Tensor, Xe : Tensor, y : Tensor):
        pass

    def predict(self, Xc : Tensor, Xe : Tensor) -> Tensor:
        pass

    def sample_y(self, Xc : Tensor, Xe : Tensor) -> Tensor:
        pass

    def sample_f(self):
        raise RuntimeError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def support_ts(self) -> bool:
        return False

    @property
    def support_grad(self) -> bool:
        return True

    def save(self, save_dir : str):
        pass

    def load(self, load_dir : str):
        pass
