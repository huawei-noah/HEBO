import pytest
import numpy as np
import pandas as pd
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

def obj(x : pd.DataFrame) -> np.ndarray:
    return sum(x[f'x{i}'].values.astype(float).reshape(-1, 1) ** 2 for i in range(4))

def obj_mixed(x : pd.DataFrame) -> np.ndarray:
    a_bonus_term = (x['x1'] == "a").values.astype(float).reshape(-1, 1)
    b_bonus_term = (x['x3'] == "b").values.astype(float).reshape(-1, 1)

    return x['x0'].values.astype(float).reshape(-1, 1) ** 2 + x['x2'].values.astype(float).reshape(-1, 1) ** 2 - a_bonus_term + b_bonus_term


@pytest.mark.parametrize('model_name', ['gp']) 
@pytest.mark.parametrize('opt_cls', [HEBO], ids = ['hebo'])
def test_opt_cont(model_name, opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x2', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x3', 'type' : 'num', 'lb' : -3, 'ub' : 7}
        ])
    model_config = {
        "rd": True,
        "E": 0.2
    }
    opt = opt_cls(space, rand_sample = 8, model_name = model_name, model_config=model_config)
    num_suggest = 0
    for i in range(9):
        num_suggest = 1
        rec = opt.suggest(n_suggestions = num_suggest)
        y   = obj(rec)
        if y.shape[0] > 1 and i > 0:
            y[np.argmax(y.reshape(-1))] = np.inf
        opt.observe(rec, y)
        num_suggest += rec.shape[0]

@pytest.mark.parametrize('model_name', ['gp']) 
@pytest.mark.parametrize('opt_cls', [HEBO], ids = ['hebo'])
def test_opt_mixed(model_name, opt_cls):
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'cat', 'categories' : ['a', 'b', 'c']},
        {'name' : 'x2', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x3', 'type' : 'cat', 'categories' : ['a', 'b', 'c', 'd']}
        ])
    model_config = {
        "rd": True,
        "E": 0.2
    }
    opt = opt_cls(space, rand_sample = 8, model_name = model_name, model_config=model_config)
    num_suggest = 0
    for i in range(9):
        num_suggest = 1
        rec = opt.suggest(n_suggestions = num_suggest)
        y   = obj_mixed(rec)
        if y.shape[0] > 1 and i > 0:
            y[np.argmax(y.reshape(-1))] = np.inf
        opt.observe(rec, y)
        num_suggest += rec.shape[0]
