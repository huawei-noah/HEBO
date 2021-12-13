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
from math import ceil

def get_bit(n, nb) -> [int]:
    ret = [int(i) for i in bin(n)[2:].zfill(nb)]
    assert len(ret) == nb
    return ret

def bin2dec(n : [int]) -> int:
    return int(''.join([str(x) for x in n]), 2)

class Lattice(nn.Module):
    def __init__(self, in_features, out_features, mono_matrix = None):
        """
        mono_matrix: out_features * in_features matirx, mono_matrix[i, j] = 1 means j-th feature for i-th output is monotonic
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mono_matrix  = mono_matrix if mono_matrix is not None else torch.zeros(out_features, in_features)
        self.weight       = nn.Parameter(torch.randn(self.out_features, 2**self.in_features) / self.in_features)
        self.bias         = nn.Parameter(torch.zeros((self.out_features)))
        self.fbits        = torch.BoolTensor(np.array([get_bit(i, self.in_features) for i in range(2**self.in_features)]).astype(bool).T)
        self.ineq_matrix  = self.get_ineq_matrix()
        assert self.mono_matrix.shape[0] == out_features
        assert self.mono_matrix.shape[1] == in_features

    def get_ineq_matrix(self) -> [torch.FloatTensor]:
        ineq_matrix = []
        for i in range(self.out_features):
            if self.mono_matrix[i].sum() > 0:
                m = []
                mono_dims = torch.where(self.mono_matrix[i])[0]
                for dim in mono_dims:
                    d0 = self.fbits[:, ~self.fbits[dim]]
                    d1 = self.fbits[:, self.fbits[dim]]
                    for k in range(2**(self.in_features - 1)):
                        idx0    = bin2dec(d0.t()[k].int().numpy().tolist())
                        idx1    = bin2dec(d1.t()[k].int().numpy().tolist())
                        c       = torch.zeros(2**self.in_features)
                        c[idx0] = 1.0
                        c[idx1] = -1.0
                        m.append(c)
                m = torch.stack(m, dim = 0)
            else:
                m = torch.zeros(0, 2**self.in_features)
            ineq_matrix.append(m)
        return ineq_matrix

    def ineq_reg(self):
        reg = 0.
        for i in range(self.out_features):
            m    = self.ineq_matrix[i]
            reg += m.mv(self.weight[i]).clamp(min = 0.).sum() # m.mv(w) < 0
        return reg

    def construct_features(self, x):
        # # TODO: this function can be refactored to be faster
        # features = [1-x[:, 0], x[:, 0]]
        # for i in range(1, self.in_features):
        #     new_features = []
        #     for f in features:
        #         new_features.append(f * (1 - x[:, i]))
        #         new_features.append(f * x[:, i])
        #     features = new_features
        # return torch.stack(features, dim = 1)
        
        # NOTE: This has O(D*2^D) complexity, but runs faster in practice as
        #       it's more cache-friendly
        features = torch.ones(x.shape[0], 2**x.shape[1])
        for i in range(self.in_features):
            features[:, self.fbits[i]]  *= x[:, i].view(-1, 1)
            features[:, ~self.fbits[i]] *= (1 - x[:, i]).view(-1, 1)
        return features

    def forward(self, x):
        f = self.construct_features(x)
        return F.linear(f, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features = {self.in_features}, out_features = {self.out_features}'
