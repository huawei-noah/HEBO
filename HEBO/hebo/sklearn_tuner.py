# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from typing import Callable

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

import warnings
warnings.filterwarnings('ignore')

def sklearn_tuner(
        model_class, 
        space_config : [dict], 
        X : np.ndarray,
        y : np.ndarray,
        metric : Callable, 
        greater_is_better : bool = True, 
        cv       = None,
        max_iter = 16, 
        report   = False,
        hebo_cfg = None, 
        verbose  = True, 
        ) -> (dict, pd.DataFrame):
    """Tuning sklearn estimator

    Parameters:
    -------------------
    model_class: class of sklearn estimator
    space_config: list of dict, specifying search space
    X, y: data used to for cross-valiation
    metrics: metric function in sklearn.metrics
    greater_is_better: whether a larger metric value is better
    cv: the 'cv' parameter in `cross_val_predict`
    max_iter: number of trials

    Returns:
    -------------------
    Best hyper-parameters and all visited data


    Example:
    -------------------
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from hebo.sklearn_tuner import sklearn_tuner

    space_cfg = [
            {'name' : 'max_depth',        'type' : 'int', 'lb' : 1, 'ub' : 20},
            {'name' : 'min_samples_leaf', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 0.5},
            {'name' : 'max_features',     'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
            {'name' : 'bootstrap',        'type' : 'bool'},
            {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
            ]
    X, y   = load_boston(return_X_y = True)
    result = sklearn_tuner(RandomForestRegressor, space_cfg, X, y, metric = r2_score, max_iter = 16)
    """
    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_config)
    opt   = HEBO(space, **hebo_cfg)
    if cv is None:
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    for i in range(max_iter):
        rec     = opt.suggest()
        hyp     = rec.iloc[0].to_dict()
        for k in hyp:
            if space.paras[k].is_numeric and space.paras[k].is_discrete:
                hyp[k] = int(hyp[k])
        model   = model_class(**hyp)
        pred    = cross_val_predict(model, X, y, cv = cv)
        score_v = metric(y, pred)
        sign    = -1. if greater_is_better else 1.0
        opt.observe(rec, np.array([sign * score_v]))
        if verbose:
            print('Iter %d, best metric: %g' % (i, sign * opt.y.min()), flush = True)
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
    if report:
        return best_hyp.to_dict(), df_report
    return best_hyp.to_dict()
