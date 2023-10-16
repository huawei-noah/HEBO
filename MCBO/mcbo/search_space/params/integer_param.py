# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import math

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


class IntegerPara(Parameter):
    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        self.lb = math.ceil(param_dict.get('lb'))
        self.ub = int(param_dict.get('ub'))

    def sample(self, num=1):
        assert (num > 0)
        return np.random.randint(self.lb, self.ub + 1, num)

    def transform(self, x):
        # Normalise
        normalised_x = (x.astype(float) - self.lb) / (self.ub - self.lb)
        return torch.tensor(normalised_x, dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Un-normalise
        x = (self.ub - self.lb) * x.clip(0, 1) + self.lb
        return x.round().astype(int)

    @property
    def is_disc(self) -> bool:
        return True

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
    def transfo_lb(self) -> float:
        return -.5

    @property
    def transfo_ub(self) -> float:
        return 1.5