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

from pytest import approx
from hebo.models.gp.gp import GP

def test_noise_free():
    X = torch.randn(10, 3)
    y = (X**2).sum(dim = 1).view(-1, 1) + 0.01 * torch.randn(10, 1)
    model = GP(3, 0, 1, noise_free = True, num_epochs = 10)
    model.fit(X, None, y)
    assert(approx(model.noise.item()) == 1.1 * model.noise_lb * model.yscaler.std**2)

@pytest.mark.xfail
def test_pred_likeli():
    X = torch.randn(10, 3)
    y = (X**2).sum(dim = 1).view(-1, 1) + 0.01 * torch.randn(10, 1)
    model = GP(3, 0, 1, noise_free = True, num_epochs = 10, pred_likeli = False)
    model.fit(X, None, y)

    with torch.no_grad():
        py, ps2 = model.predict(X, None)
        samp    = model.sample_y(X, None)

        model.pred_likeli = False

        py_no_noise, ps2_no_noise = model.predict(X, None)
        samp_no_noise = model.sample_y(X, None)
        assert((py_no_noise == py).all())

        delta = ps2 - ps2_no_noise
        assert((delta > 0).all())

        assert((samp != samp_no_noise).all())

def test_verbose(capsys):
    X = torch.randn(10, 3)
    y = (X**2).sum(dim = 1).view(-1, 1) + 0.01 * torch.randn(10, 1)
    model = GP(3, 0, 1, num_epochs = 10, verbose = True)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert('After'  in out)
    assert('epochs' in out)
    assert('loss'   in out)
    assert(err == '')

    model = GP(3, 0, 1, num_epochs = 10, verbose = False)
    model.fit(X, None, y)

    out, err = capsys.readouterr()
    assert(out == '')
    assert(err == '')
