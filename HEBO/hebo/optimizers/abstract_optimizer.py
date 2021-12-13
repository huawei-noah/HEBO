# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from hebo.design_space.design_space import DesignSpace

class AbstractOptimizer(ABC):
    support_parallel_opt    = False
    support_constraint      = False
    support_multi_objective = False
    support_combinatorial   = False
    support_contextual      = False
    def __init__(self, space : DesignSpace):
        self.space = space

    @abstractmethod
    def suggest(self, n_suggestions = 1, fix_input : dict = None):
        """
        Perform optimisation and give recommendation using data observed so far
        ---------------------
        n_suggestions:  number of recommendations in this iteration

        fix_input:      parameters NOT to be optimized, but rather fixed, this
                        can be used for contextual BO, where you can set the contex as a design
                        parameter and fix it during optimisation
        """
        pass

    @abstractmethod
    def observe(self, x : pd.DataFrame, y : np.ndarray):
        """
        Observe new data
        """
        pass

    @property
    @abstractmethod
    def best_x(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def best_y(self) -> float:
        pass
