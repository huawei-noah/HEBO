# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest
import torch
import torch.nn as nn
from hebo.models.scalers import TorchStandardScaler, TorchMinMaxScaler, TorchIdentityScaler

@pytest.mark.parametrize('scaler',
        [TorchIdentityScaler(), TorchStandardScaler(), TorchMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_identity(scaler):
    x      = torch.randn(10, 3) * 5 + 4
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert(res.sum() < 1e-4)

@pytest.mark.parametrize('scaler',
        [TorchIdentityScaler(), TorchStandardScaler(), TorchMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_identity_same(scaler):
    x      = torch.ones(10, 1) * 5 + 4
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert(res.sum() < 1e-4)

    xrand  = torch.randn(10, 3)
    res    = (xrand - scaler.inverse_transform(scaler.transform(xrand))).abs()
    assert(res.sum() < 1e-4)

def test_standard_scaler():
    x      = torch.randn(10, 3) * 5 + 4
    scaler = TorchStandardScaler().fit(x)
    x_tr   = scaler.transform(x)
    assert((x_tr.numpy().mean(axis = 0) - 0).sum() < 1e-6)
    assert((x_tr.numpy().std(axis = 0)  - 1).sum() < 1e-6)

def test_min_max_scaler():
    x      = torch.randn(10, 3) * 5 + 4
    scaler = TorchMinMaxScaler().fit(x)
    x_tr   = scaler.transform(x)
    assert((x_tr.numpy().min(axis = 0) - scaler.range_lb).sum() < 1e-6)
    assert((x_tr.numpy().max(axis = 0) - scaler.range_ub).sum() < 1e-6)

def test_min_max_scaler_same():
    x      = torch.ones(10, 3) * 5 + 4
    scaler = TorchMinMaxScaler().fit(x)
    x_tr   = scaler.transform(x)
    assert((x_tr.numpy().max(axis = 0) - scaler.range_ub).sum() < 1e-6)

@pytest.mark.parametrize('scaler',
        [TorchIdentityScaler(), TorchStandardScaler(), TorchMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_one_sample(scaler):
    x      = torch.zeros(1, 1)
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert(res.sum() < 1e-4)

def test_grad():
    x = torch.randn(10, 3) * 5 + 4
    x.requires_grad = True

    scaler0 = TorchStandardScaler().fit(x)
    scaler1 = TorchMinMaxScaler().fit(x)
    scaler2 = TorchIdentityScaler().fit(x)

    model = nn.Sequential(
            scaler0, 
            scaler1, 
            scaler2, 
            nn.Linear(3, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1))
    y = model(x).sum()
    y.backward()
    assert(x.grad is not None)
    assert(x.grad.shape == x.shape)
    assert(scaler0.mean.grad is None)
    assert(scaler0.std.grad is None)
    assert(scaler1.scale_.grad is None)
    assert(scaler1.min_.grad is None)
