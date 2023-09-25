# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import (AdditiveKernel, MaternKernel, ProductKernel,
                              ScaleKernel)
from gpytorch.priors import GammaPrior
from torch import FloatTensor, LongTensor

from ..layers import EmbTransform
from ..util import get_random_graph


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
    
def default_kern_rd(x, xe, y, total_dim = None, ard_kernel = True, fe = None, max_x = 1000, E=0.2):
    '''
    Get a default kernel with random decompositons. 0 <= E <=1 specifies random tree conectivity.
    '''
    kernels = []
    random_graph = get_random_graph(total_dim, E)
    for clique in random_graph:
        if fe is None:
            num_dims  = tuple(dim for dim in clique if dim < x.shape[1])
            enum_dims = tuple(dim for dim in clique if x.shape[1] <= dim < total_dim)
            clique_kernels = []
            if len(num_dims) > 0:
                ard_num_dims = len(num_dims) if ard_kernel else None
                num_kernel       = MaternKernel(nu = 1.5, ard_num_dims = ard_num_dims, active_dims = num_dims)
                if ard_kernel:
                    lscales = num_kernel.lengthscale.detach().clone().view(1, -1)
                    if len(num_dims) > 1 :
                        for dim_no, dim_name in enumerate(num_dims):
                            idx = np.random.choice(num_dims, min(len(num_dims), max_x), replace = False)
                            lscales[0, dim_no] = torch.pdist(x[idx, dim_name].view(-1, 1)).median().clamp(min = 0.02)
                    num_kernel.lengthscale = lscales
                clique_kernels.append(num_kernel)
            if len(enum_dims) > 0:
                enum_kernel = MaternKernel(nu = 1.5, active_dims = enum_dims)
                clique_kernels.append(enum_kernel)
            
            kernel = ScaleKernel(ProductKernel(*clique_kernels), outputscale_prior = GammaPrior(0.5, 0.5))
        else:
            if ard_kernel:
                kernel = ScaleKernel(MaternKernel(nu = 1.5, ard_num_dims = total_dim, active_dims=tuple(clique)))
            else:
                kernel = ScaleKernel(MaternKernel(nu = 1.5, active_dims=tuple(clique)))
            
        kernels.append(kernel)

    final_kern = ScaleKernel(AdditiveKernel(*kernels), outputscale_prior = GammaPrior(0.5, 0.5))
    final_kern.outputscale = y[torch.isfinite(y)].var()
    return final_kern
