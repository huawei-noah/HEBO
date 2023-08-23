# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


class NominalPara(Parameter):
    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        self.categories = list(param_dict.get('categories'))
        try:
            self._categories_dict = {k: v for v, k in enumerate(self.categories)}
        except TypeError:  # there are unhashable types
            self._categories_dict = None
        self.lb = 0
        self.ub = len(self.categories) - 1

    def sample(self, num=1):
        assert (num > 0)
        return np.random.choice(self.categories, num, replace=True)

    def transform(self, x: np.ndarray):
        if self._categories_dict:
            # if all objects are hashable, we can use a dict instead for faster transform
            ret = np.array(list(map(lambda a: self._categories_dict[a], x)))
        else:
            # otherwise, we fall back to searching in an array
            ret = np.array(list(map(lambda a: np.where(self.categories == a)[0][0], x)))
        return torch.tensor(ret, dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return np.array([self.categories[x_] for x_ in x.round().astype(int)])

    @property
    def is_disc(self) -> bool:
        return False

    @property
    def is_cont(self) -> bool:
        return False

    @property
    def is_nominal(self) -> bool:
        return True

    @property
    def is_ordinal(self) -> bool:
        return False

    @property
    def is_permutation(self) -> bool:
        return False

    @property
    def is_disc_after_transform(self) -> bool:
        return True

    @property
    def opt_lb(self):
        return float(self.lb)

    @property
    def opt_ub(self):
        return float(self.ub)

    @property
    def num_uniqs(self):
        return len(self.categories)

    @property
    def transfo_lb(self) -> float:
        return -.5

    @property
    def transfo_ub(self) -> float:
        return self.num_uniqs - .5