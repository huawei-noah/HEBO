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

import numpy as np
import pandas as pd
import torch

from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.util import parse_space_from_bayesmark

from hebo.design_space.design_space      import DesignSpace
from hebo.design_space.numeric_param     import NumericPara
from hebo.design_space.integer_param     import IntegerPara
from hebo.design_space.pow_param         import PowPara
from hebo.design_space.categorical_param import CategoricalPara
from hebo.design_space.bool_param        import BoolPara

def obj(x : pd.DataFrame) -> np.ndarray:
    return x['x0'].values.astype(float).reshape(-1, 1) ** 2

@pytest.mark.parametrize('model_name', ['gp', 'gpy', 'rf', 'deep_ensemble', 'gpy_mlp']) 
@pytest.mark.parametrize('opt_cls', [BO, HEBO, GeneralBO], ids = ['bo', 'hebo', 'general'])
def test_opt(model_name, opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}
        ])
    opt = opt_cls(space, rand_sample = 10, model_name = model_name)
    for i in range(11):
        num_suggest = 8 if opt.support_parallel_opt else 1
        rec = opt.suggest(n_suggestions = num_suggest)
        y   = obj(rec)
        if y.shape[0] > 1 and i > 0:
            y[np.argmax(y.reshape(-1))] = np.inf
        opt.observe(rec, y)
        if opt.y.shape[0] > 11:
            break

@pytest.mark.parametrize('opt_cls', [BO, HEBO, GeneralBO], ids = ['bo', 'hebo', 'general'])
def test_contextual_opt(opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'int', 'lb' : 0, 'ub' : 7},
        {'name' : 'x1', 'type' : 'int', 'lb' : 0, 'ub' : 7},
        ])
    opt = opt_cls(space, rand_sample = 2, model_name = 'rf')
    for i in range(2):
        n_suggestions = 8 if opt.support_parallel_opt else 1
        rec = opt.suggest(n_suggestions = n_suggestions, fix_input = {'x0' : 3})
        y   = (rec[['x0', 'x1']].values ** 2).sum(axis = 1, keepdims =True)
        assert((rec['x0'] == 3).all())
        opt.observe(rec, y)

def test_bayesmark_parser():
    api_config = \
            {'hidden_layer_sizes': {'type': 'int', 'space': 'linear', 'range': (50, 200)},
                    'alpha': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e1)},
                    'batch_size': {'type': 'int', 'space': 'linear', 'range': (10, 250)},
                    'learning_rate_init': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e-1)},
                    'tol': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e-1)},
                    'validation_fraction': {'type': 'real', 'space': 'logit', 'range': (0.1, 0.9)},
                    'beta_1': {'type': 'real', 'space': 'logit', 'range': (0.5, 0.99)},
                    'beta_2': {'type': 'real', 'space': 'logit', 'range': (0.9, 1.0 - 1e-6)},
                    'epsilon': {'type': 'real', 'space': 'log', 'range': (1e-9, 1e-6)}, 
                    'activation'  : {'type' : 'cat', 'values' : ['sigmoid', 'tanh', 'relu']}, 
                    'use_dropout'  : {'type' : 'bool'}, 
                    'dummy': {'type': 'real', 'space': 'linear', 'range': (4.1, 9.2)}}
    space = parse_space_from_bayesmark(api_config)
    assert(isinstance(space.paras['hidden_layer_sizes'], IntegerPara))
    assert(isinstance(space.paras['alpha'], PowPara))
    assert(isinstance(space.paras['batch_size'], IntegerPara))
    assert(isinstance(space.paras['learning_rate_init'], PowPara))
    assert(isinstance(space.paras['tol'], PowPara))
    assert(isinstance(space.paras['validation_fraction'], PowPara))
    assert(isinstance(space.paras['beta_1'], PowPara))
    assert(isinstance(space.paras['beta_2'], PowPara))
    assert(isinstance(space.paras['epsilon'], PowPara))
    assert(isinstance(space.paras['activation'], CategoricalPara))
    assert(isinstance(space.paras['use_dropout'], BoolPara))

    with pytest.raises(AssertionError):
        api_config = {'hidden_layer_sizes': {'type': '_int', 'space': 'linear', 'range': (50, 200)}}
        space      = parse_space_from_bayesmark(api_config)
