# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


class PermutationPara(Parameter):
    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        self.length = param_dict.get('length')
        self.repair_func = param_dict.get('repair_func', None)

    def sample(self, num=1):
        assert (num > 0)
        X = np.zeros((num, self.length), dtype=int)
        for i in range(num):
            X[i] = np.random.permutation(self.length)
        if self.repair_func is not None:
            X = self.repair_func(X)
        return X

    def transform(self, x: np.ndarray):
        return torch.tensor(x, dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        x = x.cpu().numpy()
        return x

    @property
    def is_disc(self) -> bool:
        return False

    @property
    def is_cont(self) -> bool:
        return False

    @property
    def is_nominal(self) -> bool:
        return False

    @property
    def is_ordinal(self) -> bool:
        return False

    @property
    def is_permutation(self) -> bool:
        return True

    @property
    def is_disc_after_transform(self) -> bool:
        return False

    @property
    def opt_lb(self):
        raise NotImplementedError

    @property
    def opt_ub(self):
        raise NotImplementedError

    @property
    def transf_lb(self):
        raise NotImplementedError

    @property
    def transf_ub(self):
        raise NotImplementedError
