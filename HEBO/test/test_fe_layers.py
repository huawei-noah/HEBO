# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

import pytest
from pytest import approx
import torch
from hebo.models.nn.fe_layers import ConcreteLayer, HardConcreteLayer, StochasticGate

@pytest.mark.parametrize('layer_cls', 
        [ConcreteLayer, HardConcreteLayer, StochasticGate], 
        ids = ['concrete', 'hard-concrete', 'stg']
        )
def test_layers(layer_cls):
    layer = layer_cls(3)
    x     = torch.ones(10000, 3)
    y     = layer(x)
    m     = layer.get_mask(x)
    norm  = layer.mask_norm
    assert y.mean() == approx(0.5, abs = 0.1)
    assert m.mean() == approx(0.5, abs = 0.1)

    if isinstance(layer, ConcreteLayer):
        assert norm == approx(1.5, abs = 0.1)

    layer.eval()
    y = layer(x)
    assert torch.isfinite(y).all()

@pytest.mark.parametrize('layer_cls', 
        [ConcreteLayer, HardConcreteLayer], 
        ids = ['concrete', 'hard-concrete']
        )
@pytest.mark.parametrize('temperature', [1e-6, 1e6], ids = ['low-temp', 'high-temp'])
def test_layers(layer_cls, temperature):
    layer = layer_cls(3, temperature = temperature)
    x     = torch.ones(100, 3)
    y     = layer(x)
    m     = layer.get_mask(x)

    if temperature < 0.1:
        cmp1 = m < 0.01
        cmp2 = m > 0.99
        assert (cmp1 | cmp2).all()
    elif temperature > 100:
        assert ((m - 0.5).abs() < 0.1).all()
