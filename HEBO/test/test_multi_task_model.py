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
from torch import FloatTensor

from hebo.models.model_factory import get_model, model_dict, get_model_class
from .util import check_prediction

def rand_dataset(num_out : int) -> (FloatTensor, FloatTensor):
    X = torch.randn(10, 2)
    y = torch.zeros(10, num_out)
    for i in range(num_out):
        w = torch.randn(2)
        y[:, i] = (X * w).sum(axis = 1)
    return X, y

def rand_dataset_with_enum(num_out : int) -> (FloatTensor, FloatTensor):
    X  = torch.randn(10, 2)
    Xe = torch.randint(2, (10, 2))
    y  = torch.zeros(10, num_out)
    for i in range(num_out):
        w = torch.randn(2)
        y[:, i] = (X * w).sum(axis = 1)
    return X, Xe, y

def gather_mo_models():
    model_list = []
    for name in model_dict:
        cls = get_model_class(name)
        if cls.support_multi_output:
            model_list.append(name)
    return model_list       

@pytest.fixture(params = gather_mo_models())
def model_name(request):
    return request.param

@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_init(model_name, num_out):
    model = get_model(model_name, 2, 0, num_out)
    assert(model.num_out == num_out)

@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_fit(model_name, num_out):
    """
    Model should be able to overfit simple linear model with training R2 > 0.8
    """
    model = get_model(model_name, 2, 0, num_out, num_epochs = 20)
    X, y  = rand_dataset(num_out)
    model.fit(X, None, y)
    with torch.no_grad():
        py, ps2 = model.predict(X, None)
        check_prediction(y, py, ps2)

@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_fit_with_enum(model_name, num_out):
    """
    Model should be able to overfit simple linear model with training R2 > 0.8
    """
    model    = get_model(model_name, 2, 2, num_out, num_uniqs = [2, 2], num_epochs = 20)
    X, Xe, y = rand_dataset_with_enum(num_out)
    model.fit(X, Xe, y)
    with torch.no_grad():
        py, ps2 = model.predict(X, Xe)
        check_prediction(y, py, ps2)

@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_grad(model_name, num_out):
    model = get_model(model_name, 2, 0, num_out, num_epochs = 2)
    if model.support_grad:
        X, y  = rand_dataset(num_out)
        model.fit(X, None, y)
        X.requires_grad = True
        py, ps2 = model.predict(X, None)
        ps      = ps2.sqrt()
        y = (py + ps).sum()
        y.backward()
        assert(X.grad.shape == X.shape)

        delta = 0.001
        Xdelta = X.detach().clone()
        Xdelta[0, 0] += delta
        py, ps2 = model.predict(Xdelta, None)
        ydelta  = (py + ps2.sqrt()).sum()
        grad_delta = ((ydelta - y) / delta).detach()
        assert(grad_delta.numpy() == approx(X.grad[0, 0].numpy(), abs = 0.2))

@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_sample(model_name, num_out):
    model = get_model(model_name, 2, 0, num_out, num_epochs = 1)
    X, y = rand_dataset(num_out)
    model.fit(X, None, y)
    assert(model.sample_y(X, None, 5).shape == torch.Size([5, X.shape[0], y.shape[1]]))
    
@pytest.mark.parametrize('num_out', [1, 2], ids = ['single-out', 'multi-out'])
def test_noise(model_name, num_out):
    model = get_model(model_name, 2, 0, num_out, num_epochs = 5)
    X, y = rand_dataset(num_out)
    model.fit(X, None, y)
    assert(model.noise.shape == torch.Size([model.num_out]))
    assert((model.noise >= 0).all())
