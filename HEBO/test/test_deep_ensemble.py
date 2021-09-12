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

from hebo.models.nn.deep_ensemble import BaseNet, DeepEnsemble
from sklearn.metrics import r2_score

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
        
        opt = torch.optim.Adam(net.parameters(), lr = 1e-3)
        for i in range(5):
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
def test_deep_ens(output_noise, rand_prior, num_processes):
    xc = torch.randn(16, 1)
    y  = xc ** 2
    model = DeepEnsemble(1, 0, 1, 
            output_noise  = output_noise,
            rand_prior    = rand_prior,
            num_processes = num_processes,
            l1            = 0., 
            num_epoch     = 16)
    model.fit(xc, None, y)
    with torch.no_grad():
        py, _ = model.predict(xc, None)
        assert(r2_score(y.numpy(), py.numpy()) > 0.5)

def test_verbose(capsys):
    X = torch.randn(10, 3)
    y = (X**2).sum(dim = 1).view(-1, 1) + 0.01 * torch.randn(10, 1)
    model = DeepEnsemble(3, 0, 1, num_epochs = 10, verbose = True)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert('Epoch'  in out)
    assert('loss'   in out)
    assert(err == '')

    model = DeepEnsemble(3, 0, 1, num_epochs = 10, verbose = False)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert(out == '')
    assert(err == '')
