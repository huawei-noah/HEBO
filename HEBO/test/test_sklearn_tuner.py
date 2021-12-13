# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from hebo.sklearn_tuner import sklearn_tuner


@pytest.mark.parametrize('report', [True, False], ids = ['report', 'no-report'])
def test_sklearn_tuner(report):
    space_cfg = [
            {'name' : 'max_depth',        'type' : 'int', 'lb' : 1, 'ub' : 20},
            {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1e-4, 'ub' : 0.5},
            {'name' : 'max_features',     'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
            {'name' : 'bootstrap',        'type' : 'bool'},
            {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
            ]
    X, y = load_boston(return_X_y = True)
    _    = sklearn_tuner(RandomForestRegressor, space_cfg, X, y, metric = r2_score, max_iter = 1, report = report)
