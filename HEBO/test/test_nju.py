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

from hebo.optimizers.bo import BO
from hebo.optimizers.nomr import NoMR_BO, AbsEtaDifference
from hebo.design_space      import DesignSpace

def obj(x : pd.DataFrame) -> np.ndarray:
    return x['x0'].values.astype(float).reshape(-1, 1) ** 2

def test_opt():
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : -3, 'ub' : 7},
        {'name' : 'x1', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}
        ])
    opt = NoMR_BO(space)
    for i in range(11):
        rec = opt.suggest()
        y   = obj(rec)
        opt.observe(rec, y)

    opt = NoMR_BO(
            space, 
            opt2 = BO(space, acq_cls = AbsEtaDifference, acq_conf = {'eta' : 0.5})
            )
    for i in range(11):
        rec = opt.suggest()
        y   = obj(rec)
        opt.observe(rec, y)
