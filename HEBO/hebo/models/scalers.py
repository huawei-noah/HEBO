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
        self.range_lb = float(range[0])
        self.range_ub = float(range[1])
        assert (self.range_ub > self.range_lb )

        self.scale_ = None
        self.min_   = None
        self.fitted = False

    def fit(self, x : torch.FloatTensor):
        assert(x.dim() == 2)
        with torch.no_grad():
            scaler = MinMaxScaler((self.range_lb, self.range_ub))
            scaler.fit(x.detach().numpy())
            self.scale_ = torch.FloatTensor(scaler.scale_)
            self.min_   = torch.FloatTensor(scaler.min_)
            self.fitted = True
        return self

    def forward(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.transform(x)

    def transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.scale_ * x + self.min_

    def inverse_transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return (x - self.min_) / self.scale_
