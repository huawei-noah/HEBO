# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from sklearn.preprocessing import LabelEncoder
from .param import Parameter

class BoolPara(Parameter):
    def __init__(self, param):
        super().__init__(param)
        self.lb         = 0
        self.ub         = 1

    def sample(self, num = 1):
        assert(num > 0)
        return np.random.choice([True, False], num, replace = True)

    def transform(self, x):
        return x.astype(float)

    def inverse_transform(self, x):
        return x > 0.5

    @property
    def is_numeric(self):
        # XXX: It's OK to view boolean as numeric value, this may reduce
        # dimensions if catecorical variables are procecessed via one-hot or
        # embedding
        return True 

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub
