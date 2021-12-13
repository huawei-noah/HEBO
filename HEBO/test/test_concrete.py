# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

import pytest
import numpy as np
import torch
import torch.nn as nn

from hebo.models.nn.fe_deep_ensemble import FeNet, FeDeepEnsemble
from hebo.models.nn.gumbel_linear import GumbelSelectionLayer, GumbelNet, GumbelDeepEnsemble
from .util import check_prediction

def test_fe_net():
    net = FeNet(2, 0, 1)
    x   = torch.randn(10, 2)
    y   = net(x, None)
    assert torch.isfinite(y).all()

    t   = torch.rand(())
    net = FeNet(2, 0, 1, temperature = t)
    assert net.feature_select.temperature == t

@pytest.mark.parametrize('fe_layer', ['concrete', 'hard_concrete', 'stg'])
def test_fe_ensemble(fe_layer):
    x     = torch.randn(300, 10)
    y     = x[:, 0].view(-1, 1)
    model = FeDeepEnsemble(x.shape[1], 0, 1, 
                    output_noise  = False,
                    num_ensembles = 1,
                    num_processes = 1,
                    num_epochs    = 1, 
                    fe_layer      = fe_layer, 
                    )
    model.fit(x, None, y)
    with torch.no_grad():
        py, ps2 = model.predict(x, None)
        check_prediction(y, py, ps2)


def test_gumbel_linear():
    torch.manual_seed(42)
    x     = torch.randn(100, 5)
    with torch.no_grad():
        layer = GumbelSelectionLayer(5, 1, temperature = 1e-6)
        res   = x - layer(x)
        assert res.var(axis = 0).min() < 1e-3
    
        layer.temperature = 1e6
        res = x - layer(x)
        assert res.var(axis = 0).min() > 1e-1

def test_gumbel_net():
    net = GumbelNet(2, 0, 1, reduced_dim = 1)
    x   = torch.randn(10, 2)
    y   = net(x, None)

def test_gumbel_ensemble():
    x     = torch.randn(300, 10)
    y     = x[:, 0].view(-1, 1)
    model = GumbelDeepEnsemble(x.shape[1], 0, 1, 
                    reduced_dim = 2,
                    num_epochs  = 1)
    model.fit(x, None, y)
    assert model.models[0].reduced_dim == 2
    with torch.no_grad():
        py, ps2 = model.predict(x, None)
        check_prediction(y, py, ps2)
