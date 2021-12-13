# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os, platform
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest
import torch

from hebo.models.nn.deep_ensemble import BaseNet, DeepEnsemble
from hebo.models.nn.fe_deep_ensemble import FeDeepEnsemble
from hebo.models.nn.gumbel_linear import GumbelDeepEnsemble
from .util import check_prediction

@pytest.mark.parametrize('num_cont',  [0, 1])
@pytest.mark.parametrize('num_enum', [0, 1])
@pytest.mark.parametrize('num_out',   [1, 2])
@pytest.mark.parametrize('output_noise', [True, False])
@pytest.mark.parametrize('rand_prior', [True, False])
@pytest.mark.parametrize('num_layers', [1, 2])
def test_basenet(num_cont, num_enum, num_out, output_noise, rand_prior, num_layers):
    if num_cont + num_enum > 0:
        xc  = torch.randn(10, num_cont)
        xe  = torch.randint(2, (10, num_enum))
        y   = torch.randn(10, num_out)
        net = BaseNet(num_cont, num_enum, num_out, num_uniqs = [2], output_noise = output_noise, rand_prior = rand_prior, num_layers = num_layers)
        py  = net(xc, xe)
        assert(torch.isfinite(py).all())
        assert(py.shape[0] == y.shape[0])
        assert(py.shape[1] == y.shape[1] if not output_noise else y.shape[1] * 2) 
        if output_noise:
            assert ((py[:, -1 * net.num_out:] >= 0)).all()
        
        crit = torch.nn.MSELoss()
        mse1 = crit(y, py[:, :net.num_out]).item()
        
        opt = torch.optim.Adam(net.parameters(), lr = 1e-5)
        for _ in range(10):
            py   = net(xc, xe)
            loss = crit(y, py[:, :net.num_out])
            opt.zero_grad()
            loss.backward()
            opt.step()
        mse2 = loss.item()
        assert(mse2 < mse1)

@pytest.mark.parametrize('output_noise', [True, False], ids = ['nll', 'mse'])
@pytest.mark.parametrize('rand_prior',   [True, False], ids = ['with_prior', 'no_prior'])
@pytest.mark.parametrize('num_processes', [1, 5], ids = ['seq', 'para'])
@pytest.mark.parametrize('model_cls',    [DeepEnsemble, FeDeepEnsemble, GumbelDeepEnsemble])
def test_deep_ens(output_noise, rand_prior, num_processes, model_cls):
    if platform.system() == 'Linux' and num_processes > 1:
        pytest.skip('Skip multiprocessing test in Linux')
    xc = torch.randn(16, 1)
    y  = xc ** 2
    model = model_cls(1, 0, 1, 
            output_noise  = output_noise,
            rand_prior    = rand_prior,
            num_processes = num_processes,
            l1            = 0.,
            num_epochs    = 1)
    model.fit(xc, None, y)
    assert len(model.models) == model.num_ensembles
    with torch.no_grad():
        py, ps2 = model.predict(xc, None)
        check_prediction(y, py, ps2)

def test_verbose(capsys):
    X = torch.randn(10, 3)
    y = (X**2).sum(dim = 1).view(-1, 1) + 0.01 * torch.randn(10, 1)
    model = DeepEnsemble(3, 0, 1, num_epochs = 1, verbose = True)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert('Epoch'  in out)
    assert('loss'   in out)
    assert(err == '')

    model = DeepEnsemble(3, 0, 1, num_epochs = 1, verbose = False)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert(out == '')
    assert(err == '')
