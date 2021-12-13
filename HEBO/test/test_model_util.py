# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import numpy as np
import torch
from hebo.models.util import filter_nan

def test_filter():
    x       = torch.randn(10, 1)
    y       = torch.randn(10, 3)
    y[0]    = np.nan
    y[1, 1] = np.nan

    xf, _, yf = filter_nan(x, None, y)
    assert yf.shape[0] == 9
    assert (xf == x[1:]).all()

    xf, xef, yf = filter_nan(x, None, y, keep_rule = 'all')
    assert (xf == x[2:]).all()
    assert yf.shape[0] == 8
    assert torch.isfinite(yf).all()
