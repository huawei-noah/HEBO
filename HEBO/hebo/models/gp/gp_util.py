# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


from gpytorch.kernels import MaternKernel, ScaleKernel, ProductKernel
from gpytorch.priors  import GammaPrior

from ..layers import EmbTransform

class DummyFeatureExtractor(nn.Module):
    def __init__(self, num_cont, num_enum, num_uniqs = None, emb_sizes = None):
        super().__init__()
        self.num_cont  = num_cont
        self.num_enum  = num_enum
        self.total_dim = num_cont
        if num_enum > 0:
            assert num_uniqs is not None
            self.emb_trans  = EmbTransform(num_uniqs, emb_sizes = emb_sizes)
            self.total_dim += self.emb_trans.num_out

    def forward(self, x : FloatTensor, xe : LongTensor):
        x_all = x
        if self.num_enum > 0:
            x_all = torch.cat([x, self.emb_trans(xe)], dim = 1)
        return x_all

def default_kern(x, xe, y, total_dim = None, ard_kernel = True, fe = None, max_x = 1000):
    if fe is None:
        has_num  = x  is not None and x.shape[1]  > 0
        has_enum = xe is not None and xe.shape[1] > 0
        kerns    = []
        if has_num:
            ard_num_dims = x.shape[1] if ard_kernel else None
            kernel       = MaternKernel(nu = 1.5, ard_num_dims = ard_num_dims, active_dims = torch.arange(x.shape[1]))
            if ard_kernel:
                lscales = kernel.lengthscale.detach().clone().view(1, -1)
                for i in range(x.shape[1]):
                    idx = np.random.choice(x.shape[0], min(x.shape[0], max_x), replace = False)
                    lscales[0, i] = torch.pdist(x[idx, i].view(-1, 1)).median().clamp(min = 0.02)
                kernel.lengthscale = lscales
            kerns.append(kernel)
        if has_enum:
            kernel = MaternKernel(nu = 1.5, active_dims = torch.arange(x.shape[1], total_dim))
            kerns.append(kernel)
        final_kern = ScaleKernel(ProductKernel(*kerns), outputscale_prior = GammaPrior(0.5, 0.5))
        final_kern.outputscale = y[torch.isfinite(y)].var()
        return final_kern
    else:
        if ard_kernel:
            kernel = ScaleKernel(MaternKernel(nu = 1.5, ard_num_dims = total_dim))
        else:
            kernel = ScaleKernel(MaternKernel(nu = 1.5))
        kernel.outputscale = y[torch.isfinite(y)].var()
        return kernel
