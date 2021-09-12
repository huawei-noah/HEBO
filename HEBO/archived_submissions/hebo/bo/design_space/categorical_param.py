# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from sklearn.preprocessing import LabelEncoder
from .param import Parameter

class CategoricalPara(Parameter):
    def __init__(self, param):
        super().__init__(param)
        self.categories = list(map(str, param['categories']))
        self.encoder    = LabelEncoder().fit(self.categories)
        self.lb         = min(self.encoder.transform(self.categories))
        self.ub         = max(self.encoder.transform(self.categories))
        assert self.lb == 0
        assert self.ub == len(self.categories) - 1

    def sample(self, num = 1):
        assert(num > 0)
        return self.encoder.inverse_transform(np.random.randint(self.lb, self.ub + 1, num))

    def transform(self, x : np.ndarray):
        return self.encoder.transform(x.astype(str).reshape(-1))

    def inverse_transform(self, x):
        return self.encoder.inverse_transform(x.round().astype(int).reshape(-1))

    @property
    def is_numeric(self):
        return False

    @property
    def is_discrete(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub
