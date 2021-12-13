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
import numpy as np

from hebo.models.base_model import BaseModel
from hebo.models.model_factory import get_model, get_model_class, model_dict
from .util import check_prediction


@pytest.fixture(params = list(model_dict.keys()))
def model_name(request):
    return request.param

def test_is_basic_model(model_name):
    assert(issubclass(type(get_model(model_name, 1, 0, 1)), BaseModel))

def test_model_can_be_initialized(model_name):
    model1      = get_model(model_name, 1, 0, 1)
    model_class = get_model_class(model_name)
    model2      = model_class(1, 0, 1)
    assert isinstance(model1, BaseModel)
    assert isinstance(model2, BaseModel)

def test_fit_with_cont_enum(model_name):
    Xc = torch.randn(50, 1)
    Xe = torch.randint(2, (50, 1))
    y  = torch.zeros(50, 1)
    y[Xe.squeeze() == 1] = Xc[Xe.squeeze() == 1]
    y[Xe.squeeze() == 0] = -1 * Xc[Xe.squeeze() == 0]
    model = get_model(model_name, 1, 1, 1, num_uniqs = [2], num_epochs = 1)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_prediction(y, py, ps2)

def test_fit_with_cont_only(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_prediction(y, py, ps2)

def test_fit_with_enum_only(model_name):
    Xc = None
    Xe = torch.randint(2, (50, 1))
    y  = Xe.float()
    model = get_model(model_name, 0, 1, 1, num_uniqs = [2], num_epochs = 1)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))
    with torch.no_grad():
        py, ps2 = model.predict(Xc, Xe)
        check_prediction(y, py, ps2)

def test_thompson_sampling(model_name):
    model = get_model(model_name, 1, 0, 1)
    Xc = torch.randn(10, 1)
    Xe = None
    y  = Xc ** 2
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(Xc, Xe, y + 1e-2 * torch.randn(y.shape))

    with torch.no_grad():
        if model.support_ts:
            f  = model.sample_f()
            py = f(Xc, Xe)
            check_prediction(y, py)
        else:
            with pytest.raises(NotImplementedError):
                f = model.sample_f()

def test_grad(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc + 0.01 * torch.randn(50, 1)
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(Xc, Xe, y)
    if model.support_grad:
        X_tst = torch.randn(50, 1)
        X_tst.requires_grad = True
        py, _ = model.predict(X_tst, None)
        obj = py.sum()
        obj.backward()
        assert(X_tst.grad is not None)
    else:
        pytest.skip('%s does not support grad' % model_name)

def test_noise(model_name):
    Xc = torch.randn(50, 1)
    Xe = None
    y  = Xc + 0.01 * torch.randn(50, 1)
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(Xc, Xe, y)
    with torch.no_grad():
        assert(model.noise.shape == torch.Size([model.num_out]))
        assert(super(type(model), model).noise.shape == torch.Size([model.num_out]))
        assert((model.noise >= 0).all())

@pytest.mark.xfail
def test_warm_start(model_name):
    Xc    = torch.randn(64, 1)
    y     = Xc + 0.01 * torch.randn(Xc.shape[0], 1)
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    if model.support_warm_start:
        model.fit(Xc, None, y)
        with torch.no_grad():
            py, _ = model.predict(Xc, None)
            err1  = ((py - y)**2).mean()
        model.fit(Xc, None, y)
        with torch.no_grad():
            py, _ = model.predict(Xc, None)
            err2  = ((py - y)**2).mean()
        assert(err2 < err1)
    else:
        pytest.skip('%s does not support warm start' % model_name)

def test_fit_with_nan(model_name):
    x     = torch.randn(10, 1)
    y     = x + 1e-4 * torch.randn(10, 1)
    y[0]  = np.nan
    model = get_model(model_name, 1, 0, 1, num_epochs = 1)
    model.fit(x, None, y)
    with torch.no_grad():
        py, ps2 = model.predict(x, None)
        check_prediction(y, py, ps2)
