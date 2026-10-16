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
from sklearn.base    import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from hebo.sklearn_tuner import sklearn_tuner


class _TypeRecorder(BaseEstimator, RegressorMixin):
    """Records the python type of each hyper-parameter it receives at fit time."""
    seen = []

    def __init__(self, flag = True, depth = 1):
        self.flag  = flag
        self.depth = depth

    def fit(self, X, y):
        _TypeRecorder.seen.append((type(self.flag).__name__, type(self.depth).__name__))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_bool_param_stays_bool():
    # Regression test for issue #108: a `bool` parameter must reach the
    # estimator as a python bool, not be coerced to int (0/1).
    _TypeRecorder.seen = []
    space_cfg = [
        {'name' : 'flag',  'type' : 'bool'},
        {'name' : 'depth', 'type' : 'int', 'lb' : 1, 'ub' : 5},
    ]
    X, y = np.random.randn(40, 2), np.random.randn(40)
    sklearn_tuner(_TypeRecorder, space_cfg, X, y,
                  metric = r2_score, max_iter = 3, verbose = False)

    flag_types  = {t[0] for t in _TypeRecorder.seen}
    depth_types = {t[1] for t in _TypeRecorder.seen}
    assert _TypeRecorder.seen, "estimator was never fitted"
    assert flag_types  == {'bool'}, f"bool param coerced to {flag_types}, expected bool"
    assert depth_types == {'int'},  f"int param became {depth_types}, expected int"
