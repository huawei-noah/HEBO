# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import get_scorer, make_scorer
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
        n_splits = 5,
        max_iter = 16, 
        report   = False
        ) -> (dict, pd.DataFrame):
    """Tuning sklearn estimator

    Parameters:
    -------------------
    model_class: class of sklearn estimator
    space_config: list of dict, specifying search space
    X, y: data used to for cross-valiation
    metrics: metric function in sklearn.metrics
    greater_is_better: whether a larger metric value is better
    n_splits: split data into `n_splits` parts for cross validation
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
            {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1e-4, 'ub' : 0.5},
            {'name' : 'max_features',     'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
            {'name' : 'bootstrap',        'type' : 'bool'},
            {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
            ]
    X, y   = load_boston(return_X_y = True)
    result = sklearn_tuner(RandomForestRegressor, space_cfg, X, y, metric = r2_score, max_iter = 16)
    """
    space = DesignSpace().parse(space_config)
    opt   = HEBO(space)
    for i in range(max_iter):
        rec     = opt.suggest()
        model   = model_class(**rec.iloc[0].to_dict())
        pred    = cross_val_predict(model, X, y, cv = KFold(n_splits = n_splits, shuffle = True))
        score_v = metric(y, pred)
        sign    = -1. if greater_is_better else 1.0
        opt.observe(rec, np.array([sign * score_v]))
        print('Iter %d, best metric: %g' % (i, sign * opt.y.min()))
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
    if report:
        return best_hyp.to_dict(), df_report
    else:
        return best_hyp.to_dict()
