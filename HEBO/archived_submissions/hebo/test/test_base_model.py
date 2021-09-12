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
import numpy as np
from torch import FloatTensor
from sklearn.metrics import r2_score

from bo.models.base_model import BaseModel
from bo.models.model_factory import get_model, model_dict

def check_overfitted(y_true : FloatTensor, py_pred : FloatTensor, ps2_pred : FloatTensor) -> bool:
    y  = y_true.numpy().reshape(-1)
    py = py_pred.numpy().reshape(-1)
    ps = ps2_pred.sqrt().numpy().reshape(-1)
    assert r2_score(y, py) > 0.8
    assert (py + 3 * ps >= y).sum() > 0.9 * y.size
    assert (py - 3 * ps <= y).sum() > 0.9 * y.size

@pytest.fixture(params = list(model_dict.keys()))
def model_name(request):
    return request.param

def test_is_basic_model(model_name):
    assert(issubclass(type(get_model(model_name, 1, 0, 1)), BaseModel))

def test_model_can_be_initialized(model_name):
    model = get_model(model_name, 1, 0, 1)

def test_fit_with_cont_enum(model_name):
    Xc = torch.randn(50, 1)
    Xe = torch.randint(2, (50, 1))
    y  = torch.zeros(50, 1)
    y[Xe.squeeze() == 1] = Xc[Xe.squeeze() == 1]
    y[Xe.squeeze() == 0] = -1 * Xc[Xe.squeeze() == 0]
    model = get_model(model_name, 1, 1, 1, num_uniqs = [2], num_epochs = 30)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_overfitted(y, py, ps2)
        samp = model.sample_y(Xc, Xe, 50).mean(axis = 0)
        assert r2_score(y.numpy(), samp.numpy()) > 0.8

def test_fit_with_cont_only(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc
    model = get_model(model_name, 1, 0, 1, num_epochs = 30)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_overfitted(y, py, ps2)

def test_fit_with_enum_only(model_name):
    Xc = None
    Xe = torch.randint(2, (50, 1))
    y  = Xe.float()
    model = get_model(model_name, 0, 1, 1, num_uniqs = [2], num_epochs = 30)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_overfitted(y, py, ps2)

def test_thompson_sampling(model_name):
    model = get_model(model_name, 1, 0, 1)
    if model.support_ts:
        assert False, "Test cases for TS have not been concstructed"
    else:
        with pytest.raises(RuntimeError):
            f = model.sample_f()

def test_grad(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc + 0.01 * torch.randn(50, 1)
    model = get_model(model_name, 1, 0, 1, num_epochs = 30)
    model.fit(Xc, Xe, y)
    if model.support_grad:
        X_tst = torch.randn(50, 1)
        X_tst.requires_grad = True
        py, _ = model.predict(X_tst, None)
        obj = py.sum()
        obj.backward()
        assert(X_tst.grad is not None)

def test_noise(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc + 0.01 * torch.randn(50, 1)
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(Xc, Xe, y)
    with torch.no_grad():
        assert(model.noise.shape == torch.Size([model.num_out]))
        assert((model.noise >= 0).all())

def test_multi_task():
    assert True
