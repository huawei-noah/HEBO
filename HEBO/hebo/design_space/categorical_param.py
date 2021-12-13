# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from .param import Parameter

class CategoricalPara(Parameter):
    def __init__(self, param):
        super().__init__(param)
        self.categories = list(param['categories'])
        try:
            self._categories_dict = {k:v for v, k in enumerate(self.categories)}
        except TypeError: # there are unhashable types
            self._categories_dict = None
        self.lb         = 0
        self.ub         = len(self.categories) - 1

    def sample(self, num = 1):
        assert(num > 0)
        return np.random.choice(self.categories, num, replace = True)

    def transform(self, x : np.ndarray):
        if self._categories_dict:
            # if all objects are hashable, we can use a dict instead for faster transform
            ret = np.array(list(map(lambda a: self._categories_dict[a], x)))
        else:
            # otherwise, we fall back to searching in an array
            ret = np.array(list(map(lambda a: np.where(self.categories == a)[0][0], x)))
        return ret.astype(float)

    def inverse_transform(self, x):
        return np.array([self.categories[x_] for x_ in x.round().astype(int)])

    @property
    def is_numeric(self):
        return False

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

    @property
    def num_uniqs(self):
        return len(self.categories)
