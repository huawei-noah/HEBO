# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from torch import FloatTensor

def check_prediction(y_true : FloatTensor, py_pred : FloatTensor, ps2_pred : FloatTensor = None) -> bool:
    y  = y_true.numpy().reshape(-1)
    py = py_pred.numpy().reshape(-1)
    assert y.shape == py.shape
    assert np.isfinite(py).all()
    if ps2_pred is not None:
        assert (ps2_pred.numpy() > 0).all()
