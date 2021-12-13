# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from .param import Parameter

class StepIntPara(Parameter):
    """
    Integer parameter, that increments with a fixed step, like `[4, 8, 12, 16]`
    The config would be like `{'name' : 'x', 'type' : 'step_int', 'lb' : 4, 'ub' : 16, 'step' : 4}`
    """
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb       = round(param_dict['lb'])
        self.ub       = round(param_dict['ub'])
        self.step     = round(param_dict['step'])
        self.num_step = (self.ub - self.lb) // self.step
    
    def sample(self, num = 1):
        return np.random.randint(0, self.num_step + 1, num) * self.step + self.lb

    def transform(self, x : np.ndarray) -> np.ndarray:
        return (x - self.lb) / self.step

    def inverse_transform(self, x : np.ndarray) -> np.ndarray:
        x_recover = x * self.step + self.lb
        return x_recover.round().astype(int)

    @property
    def is_numeric(self):
        return True

    @property
    def opt_lb(self):
        return 0.

    @property
    def opt_ub(self):
        return 1. * self.num_step

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True
