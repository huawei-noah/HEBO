# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchIdentityScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, x : torch.FloatTensor):
        return self

    def forward(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return x

    def transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)

    def inverse_transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return x

class TorchStandardScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean   = None
        self.std    = None
        self.fitted = False

    def fit(self, x : torch.FloatTensor):
        assert(x.dim() == 2)
        with torch.no_grad():
            scaler = StandardScaler()
            scaler.fit(x.detach().numpy())

            self.mean = torch.FloatTensor(scaler.mean_.copy()).view(-1)
            self.std  = torch.FloatTensor(scaler.scale_.copy()).view(-1)
            invalid   = ~(torch.isfinite(self.mean) & torch.isfinite(self.std))
            self.mean[invalid] = 0. # somethime we face data with some all-NaN columns
            self.std[invalid]  = 1.
            return self

    def forward(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return (x - self.mean) / self.std

    def transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)

    def inverse_transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return x * self.std + self.mean

class TorchMinMaxScaler(nn.Module):
    def __init__(self, range : tuple = (0, 1)):
        super().__init__()
        self.range_lb = range[0]
        self.range_ub = range[1]
        self.data_lb  = None
        self.data_ub  = None
        self.fitted   = False

    def fit(self, x : torch.FloatTensor):
        assert(x.dim() == 2)
        with torch.no_grad():
            self.data_lb = x.min(dim = 0).values.detach().clone()
            self.data_ub = x.max(dim = 0).values.detach().clone()
            self.fitted  = True
            assert(torch.isfinite(self.data_lb).all())
            assert(torch.isfinite(self.data_ub).all())
            return self

    def to_unit(self, x, lb, ub):
        return (x - lb) / (ub - lb)

    def forward(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.to_unit(x, self.data_lb, self.data_ub) * (self.range_ub - self.range_lb) + self.range_lb

    def transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)

    def inverse_transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.to_unit(x, self.range_lb, self.range_ub) * (self.data_ub - self.data_lb) + self.data_lb
