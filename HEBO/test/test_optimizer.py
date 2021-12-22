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
import pandas as pd

from hebo.optimizers.noisy_opt import NoisyOpt
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from hebo.optimizers.vcbo import VCBO
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo_contextual import HEBO_VectorContextual
from hebo.optimizers.cmaes import CMAES
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
@pytest.mark.parametrize('opt_cls', [BO, HEBO, GeneralBO, NoisyOpt], ids = ['bo', 'hebo', 'general', 'noisy'])
def test_opt(model_name, opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}
        ])
    opt = opt_cls(space, rand_sample = 8, model_name = model_name)
    for i in range(11):
        num_suggest = 8 if opt.support_parallel_opt else 1
        rec = opt.suggest(n_suggestions = num_suggest)
        y   = obj(rec)
        if y.shape[0] > 1 and i > 0:
            y[np.argmax(y.reshape(-1))] = np.inf
        opt.observe(rec, y)
        if opt.y.shape[0] > 11:
            break

def test_vcbo():
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        ])
    opt = VCBO(space, rand_sample = 8)
    for i in range(11):
        num_suggest = 8 if opt.support_parallel_opt else 1
        rec = opt.suggest(n_suggestions = num_suggest)
        if len(opt.Y) > 11:
            break

@pytest.mark.parametrize('start_from_mu', [True, False])
def test_cmaes(start_from_mu):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}
        ])
    opt = CMAES(space)
    for i in range(2):
        rec = opt.suggest()
        y   = obj(rec)
        if y.shape[0] > 1 and i > 0:
            y[np.argmax(y.reshape(-1))] = np.inf
        opt.observe(rec, y)

    mu_old = opt.mu.clone()
    opt.restart(start_from_mu)
    assert (opt.C.diag() == 1).all()
    assert (opt.p_sigma.norm() == 0)
    assert (opt.p_c.norm() == 0)
    assert ((mu_old - opt.mu).norm() == 0) == start_from_mu

@pytest.mark.parametrize('opt_cls', [BO, HEBO, GeneralBO], ids = ['bo', 'hebo', 'general'])
def test_contextual_opt(opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'int', 'lb' : -20, 'ub' : 20},
        {'name' : 'x1', 'type' : 'int', 'lb' : -20, 'ub' : 20},
        ])
    opt = opt_cls(space, rand_sample = 2, model_name = 'rf')
    for _ in range(2):
        n_suggestions = 8 if opt.support_parallel_opt else 1
        context = np.random.randint(40) - 20
        rec = opt.suggest(n_suggestions = n_suggestions, fix_input = {'x0' : context})
        y   = (rec[['x0', 'x1']].values ** 2).sum(axis = 1, keepdims =True)
        assert((rec['x0'] == context).all())
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
    assert(isinstance(space.paras['dummy'], NumericPara))

    with pytest.raises(AssertionError):
        api_config = {'hidden_layer_sizes': {'type': '_int', 'space': 'linear', 'range': (50, 200)}}
        space      = parse_space_from_bayesmark(api_config)

@pytest.mark.parametrize('num_suggest', [1, 4, 50])
def test_general_mo_constrained_opt(num_suggest):

    def f(param : pd.DataFrame) -> np.ndarray:
        x  = param[['x0']].values
        o1 = x**2
        o2 = (x - 3)*2
        c1 = -10 - x # x > -10
        return np.hstack([o1, o2, c1])

    space       = DesignSpace().parse([{'name' : 'x0', 'type' : 'num', 'lb' : -1, 'ub' : 4.0}])
    opt         = GeneralBO(space, 2, 1, rand_sample = 4)
    opt.evo_pop = 20
    for _ in range(2):
        rec = opt.suggest(num_suggest)
        y   = f(rec)
        opt.observe(rec, y)


def test_contextual_vector():
    def obj(param : pd.DataFrame) -> np.ndarray:
        return (param[['x', 'c1', 'c2']].values**2).sum(axis = 1, keepdims = True)
    space = DesignSpace().parse([
        {'name' : 'x', 'type' : 'num', 'lb' : -1, 'ub' : 1}, 
        {'name' : 'c1', 'type' : 'num', 'lb' : -1, 'ub' : 1}, 
        {'name' : 'c2', 'type' : 'int', 'lb' : -3, 'ub' : 3}, 
        {'name' : 'c3', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}
    ])
    context_dict = {'context1' : {'c1' : 0.5, 'c2' : 1, 'c3' : 'a'}, 'context2' : {'c1' : 0.3, 'c2' : -1, 'c3' : 'b'}}
    opt = HEBO_VectorContextual(space, context_dict, rand_sample = 10, model_name = 'rf')
    for i in range(11):
        opt.context = ['context1', 'context2'][i % 2]
        rec         = opt.suggest(1)
        y           = obj(rec)
        opt.observe(rec, y)

def test_int_exponent():
    space = DesignSpace().parse([{'name' : 'x', 'type' : 'int_exponent', 'lb' : 16, 'ub' : 2048, 'base' : 2}])
    opt   = HEBO(space, rand_sample = 100)
    rec   = opt.suggest(100)
    assert np.all(np.log2(rec.values) == np.log2(rec.values).round())

@pytest.mark.parametrize('opt_cls', [BO, HEBO, CMAES], ids = ['bo', 'hebo', 'cmaes'])
def test_best_xy(opt_cls):
    space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : 0, 'ub' : 1}])
    opt   = opt_cls(space, rand_sample = 100)

    with pytest.raises(RuntimeError):
        best_x = opt.best_x
    with pytest.raises(RuntimeError):
        best_y = opt.best_y

    rec = opt.suggest()
    y   = rec['x'].values.reshape(-1, 1)
    opt.observe(rec, y)

    assert isinstance(opt.best_x, pd.DataFrame)
    assert isinstance(opt.best_y, float)
