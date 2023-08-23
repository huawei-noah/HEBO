# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


class BoolPara(Parameter):
    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        self.categories = [0, 1]
        self.lb = 0
        self.ub = 1

    def sample(self, num=1):
        assert (num > 0)
        return np.random.choice([True, False], num, replace=True)

    def transform(self, x):
        return torch.tensor(x.astype(bool), dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x > 0.5

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
    def opt_lb(self) -> float:
        return 0

    @property
    def opt_ub(self) -> float:
        return 1

    @property
    def transfo_lb(self) -> float:
        return 0

    @property
    def transfo_ub(self) -> float:
        return 1