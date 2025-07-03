# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
import numpy as np
from .param import Parameter
from torch import Tensor

class SparseGridPara(Parameter):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.values   = param_dict['values']
        self.lb   = min(self.values)
        self.ub   = max(self.values)

    def sample(self, num = 1):
        assert(num > 0)
        return np.random.choice(self.values, num)

    def transform(self, x):
        return x.astype(float)

    def inverse_transform(self, x):
        return x.round().astype(int)

    @property
    def is_numeric(self):
        return True

    @property
    def opt_lb(self):
        return float(self.lb)

    @property
    def opt_ub(self):
        return float(self.ub)

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True

    def transform_random_uniform(self, s : Tensor) -> float:
        """Take a random uniform s in [0, 1] and return a random value from the space.
        """
        return self.values[int(s * len(self.values))]

