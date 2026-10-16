# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

import importlib
import warnings

import pytest


def _head_filter_action():
    """Return the action of the highest-priority entry in warnings.filters."""
    return warnings.filters[0][0] if warnings.filters else None


def test_import_does_not_install_global_ignore_filter():
    # Regression test: importing hebo.sklearn_tuner must NOT install a
    # process-wide 'ignore' warning filter. A library silencing every
    # warning of the host program just by being imported is a side effect
    # that hides the user's own warnings.
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter('default')

        import hebo.sklearn_tuner as st
        importlib.reload(st)

        action = _head_filter_action()
        assert action != 'ignore', (
            "importing hebo.sklearn_tuner installed a global "
            "warnings 'ignore' filter, silencing all caller warnings"
        )


def test_tuning_does_not_leak_warning_filter():
    # Running the tuner must leave the caller's warning filters unchanged.
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics  import r2_score
    from hebo.sklearn_tuner import sklearn_tuner

    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter('default')
        before = list(warnings.filters)

        space_cfg = [
            {'name' : 'max_depth', 'type' : 'int', 'lb' : 1, 'ub' : 4},
        ]
        X, y = np.random.randn(30, 2), np.random.randn(30)
        sklearn_tuner(RandomForestRegressor, space_cfg, X, y,
                      metric = r2_score, max_iter = 2, verbose = False)

        after = list(warnings.filters)
        assert after == before, (
            "sklearn_tuner mutated the caller's global warnings.filters"
        )
