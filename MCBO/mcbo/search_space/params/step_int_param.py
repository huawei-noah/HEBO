# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


class StepIntPara(Parameter):
    """
    Integer parameter, that increments with a fixed step, like `[4, 8, 12, 16]`
    The config would be like `{'name' : 'x', 'type' : 'step_int', 'lb' : 4, 'ub' : 16, 'step' : 4}`
    """

    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        self.lb = round(param_dict.get('lb'))
        self.ub = round(param_dict.get('ub'))
        self.step = round(param_dict.get('step'))
        self.num_step = (self.ub - self.lb) // self.step

    def sample(self, num=1):
        return np.random.randint(0, self.num_step + 1, num) * self.step + self.lb

    def transform(self, x: np.ndarray) -> np.ndarray:
        step_num = (x - self.lb) / self.step

        # Normalise
        normalised_step_num = step_num / self.num_step

        return torch.tensor(normalised_step_num, dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        x = x.cpu().numpy()

        # Un-normalise
        x = x * self.num_step

        x_recover = x * self.step + self.lb
        return x_recover.round().astype(int)

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
        return 0.

    @property
    def opt_ub(self):
        return 1. * self.num_step
