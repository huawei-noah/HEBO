# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonoLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.w            = nn.Parameter(torch.randn(out_features, in_features) + 2)
        self.b            = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, self.w.clamp(min = 0), self.b) / (2 * self.in_features)

    def extra_repr(self):
        return f'in_features = {self.in_features}, out_features = {self.out_features}'

class PartialMonoLinear(nn.Module):
    def __init__(self, in_features, out_features, mono_idxs):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mono_idxs    = mono_idxs
        self.other_idxs   = [i for i in range(in_features) if i not in mono_idxs]
        self.num_mono     = len(mono_idxs)
        self.mono_linear  = MonoLinear(self.num_mono, out_features)
        self.linear       = nn.Linear(len(self.other_idxs), out_features)

    def forward(self, x):
        return self.linear(x[:, self.other_idxs]) + self.mono_linear(x[:, self.mono_idxs])

    def extra_repr(self):
        return f'in_features = {self.in_features}, out_features = {self.out_features}, num_mono = {self.num_mono}'

class MonoLinearAct(nn.Module):
    def __init__(self, in_features, out_features, act):
        super().__init__()
        self.linear = MonoLinear(in_features, out_features)
        self.act    = act

    def forward(self, x):
        return self.act(self.linear(x))

class MonoConvex(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = MonoLinearAct(in_features, out_features, nn.LeakyReLU())

    def forward(self, x):
        return self.layer(x)

class MonoConcave(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = MonoLinear(in_features, out_features)
    
    def forward(self, x):
        o = self.linear(x)
        return -1 * F.leaky_relu(-1 * o)

class MonoNonLinear(nn.Module):
    """
    Only used as hidden layer
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert out_features > 1
        self.convex  = MonoConvex(in_features, out_features)
        self.concave = MonoConcave(in_features, out_features)
    
    def forward(self, x):
        return 0.5 * (self.concave(x) + self.convex(x))

class KumarWarp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._a  = nn.Parameter(torch.zeros((dim)))
        self._b  = nn.Parameter(torch.zeros((dim)))
        self._al = 0.01
        self._au = 10
        self._bl = 0.01
        self._bu = 10
        self.eps = 1e-6

    @property
    def a(self):
        # bound a an b to [al, au]
        return self._al + (self._au - self._al) * torch.sigmoid(self._a)

    @property
    def b(self):
        # bound a an b to [al, au]
        return self._bl + (self._bu - self._bl) * torch.sigmoid(self._b)

    def l1reg(self):
        """
        a = 1; b = 1 for identity transformation
        """
        reg_a = (self.a - 1.0).abs().sum()
        reg_b = (self.b - 1.0).abs().sum()
        return reg_a + reg_b

    def forward(self, x):
        out =  1. - (1. - x.clamp(min = self.eps, max = 1-self.eps) ** self.a) ** self.b
        return out

class KumarWarpConcave(KumarWarp):
    def __init__(self, dim):
        super().__init__(dim)
        self._al = 0.06
        self._au = 1.0
        self._bl = 1.0
        self._bu = 8.0

class KumarWarpConvex(KumarWarp):
    def __init__(self, dim):
        super().__init__(dim)
        self._al = 1.0
        self._au = 8.0
        self._bl = 0.0
        self._bu = 1.0

class KumarWarpDiminishReturn(KumarWarp):
    def __init__(self, dim):
        super().__init__(dim)
        self._al = 0.05
        self._au = 8.0
        self._bl = 1.05
        self._bu = 8.0
