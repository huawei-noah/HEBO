# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn


def get_fe_layer(name):
    fe_layer_dict = {
            'concrete'      : ConcreteLayer,
            'hard_concrete' : HardConcreteLayer,
            'stg'           : StochasticGate
            }
    return fe_layer_dict[name]

class ConcreteLayer(nn.Module):
    def __init__(self, dim, temperature = 0.1):
        super().__init__()
        self.dim         = dim
        self.logits      = nn.Parameter(0.01 * torch.randn(dim))
        self.temperature = temperature

    def get_mask(self, x):
        if self.training: 
            mask = self.gumbel_sigmoid(logits = self.logits * torch.ones(x.shape), temperature = self.temperature)
        else:
            mask = self.gumbel_sigmoid(logits = self.logits, temperature = self.temperature)
        return mask

    @property
    def mask_norm(self):
        """
        (Expected) L0 norm of mask
        """
        return self.logits.sigmoid().sum()

    def forward(self, x):
        return x * self.get_mask(x)

    def gumbel_sigmoid(self, logits, temperature):
        eps     = torch.finfo(torch.float32).eps
        u       = torch.rand(logits.shape).clamp(eps, 1-eps)
        u_logit = torch.log(u / (1 - u))
        z_logit = logits + u_logit
        y       = (z_logit / temperature).sigmoid()
        return y

    def extra_repr(self):
        return f'dim = {self.dim}, mask_norm = {self.mask_norm}'

class HardConcreteLayer(ConcreteLayer):
    def __init__(self, dim, temperature = 0.1, lb = -0.1, ub = 1.1):
        super().__init__(dim, temperature)
        assert lb < 0.
        assert ub > 1.
        self.lb = lb
        self.ub = ub
    
    def get_mask(self, x):
        mask_concrete = super().get_mask(x)
        mask_stretch  = mask_concrete * (self.ub - self.lb) + self.lb
        mask          = mask_stretch.clamp(0., 1.)
        return mask

class StochasticGate(nn.Module):
    def __init__(self, dim, temperature = 1.0):
        super().__init__()
        self.dim         = dim
        self.mu          = nn.Parameter(0.01 * torch.randn(dim))
        self.temperature = torch.tensor(temperature)
        self.mask        = torch.ones(dim)

    @property
    def mask_norm(self):
        normed = (self.mu + 0.5) / self.temperature
        reg    = 0.5 * (1 + torch.erf(normed / np.sqrt(2)))
        return reg.sum() 

    def get_mask(self, x):
        if self.training:
            epsilon = self.temperature * torch.randn(x.shape)
            mask    = (epsilon + self.mu + 0.5).clamp(min = 0.,  max = 1.)
        else:
            epsilon = self.temperature * torch.randn(self.dim)
            mask    = (epsilon + self.mu + 0.5).clamp(min = 0.,  max = 1.)
        return mask

    def forward(self, x):
        return x * self.get_mask(x)

    def extra_repr(self):
        return f'dim = {self.dim}, mask_norm = {self.mask_norm}'
