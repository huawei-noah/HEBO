# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import numpy as np
import pandas as pd
import torch

from .abstract_optimizer import AbstractOptimizer
from .hebo import HEBO
from hebo.acquisitions.acq import Acquisition, MACE

class HEBO_VectorContextual(AbstractOptimizer):
    support_parallel_opt  = True
    support_combinatorial = True
    support_contextual    = True
    def __init__(self, 
            space,
            context_dict : dict, 
            model_name   : str = 'gpy',
            rand_sample  : int = None
            ):
        self.hebo         = HEBO(space, model_name, rand_sample)
        self.context_dict = context_dict
        self.context      = None

    @property
    def context_vector(self) -> dict: 
        fix_input = self.context_dict[self.context]
        for k in fix_input.keys():
            assert k in self.hebo.space.para_names
        return fix_input

    def suggest(self, n):
        return self.hebo.suggest(n, fix_input = self.context_vector)

    def observe(self, X, y):
        self.hebo.observe(X, y)

    @property
    def best_x(self) -> pd.DataFrame:
        raise NotImplementedError('Not supported for contextual BO')

    @property
    def best_y(self) -> pd.DataFrame:
        raise NotImplementedError('Not supported for contextual BO')
