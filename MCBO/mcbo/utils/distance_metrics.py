# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch


def hamming_distance(x1, x2, normalize=False):
    if len(x1) == 0:
        assert len(x2) == 0
        return 0
    delta = torch.abs(x1 - x2) > 1e-6

    if delta.ndim == 1:
        delta_sum = torch.sum(delta)
        if normalize:
            delta_sum = delta_sum / len(delta)
        return delta_sum
    else:
        delta_sum = torch.sum(delta, axis=1)
        if normalize:
            delta_sum = delta_sum / delta.shape[1]
        return delta_sum


def euclidean_distance(x1, x2):
    delta = (x1 - x2) ** 2
    delta = np.sqrt(delta)
    return delta


def position_distance(x1, x2):
    # position distance for permutation based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9002675
    # Note, to use numba, remove the axis variable from argsort and use a for loop to process all of the data
    x1 = torch.argsort(x1, axis=1)
    x2 = torch.argsort(x2, axis=1)

    return torch.abs(x1 - x2).sum(axis=1)
