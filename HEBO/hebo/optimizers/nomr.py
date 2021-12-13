# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

"""
Bayesian optimisation collaborated with NJU
"""

import numpy as np
import pandas as pd
import torch

from hebo.design_space import DesignSpace
from hebo.acquisitions.acq import SingleObjectiveAcq
from .abstract_optimizer import AbstractOptimizer
from .bo import BO 
from .hebo import HEBO 

class AbsEtaDifference(SingleObjectiveAcq):
    def __init__(self, model, kappa=3.0, eta=0.7, **conf):
        super().__init__(model, **conf)
        self.kappa = kappa
        self.eta = eta
        assert model.num_out == 1

    def eval(self, x: torch.FloatTensor, xe: torch.LongTensor) -> torch.FloatTensor:
        py, ps2 = self.model.predict(x, xe)
        return torch.abs(py-self.eta) - self.kappa * ps2.sqrt()

class NoMR_BO(AbstractOptimizer):
    support_parallel_opt  = False
    support_combinatorial = True
    support_contextual    = False

    def __init__(self, 
            space : DesignSpace,
            eta   : float = None,
            opt1  : AbstractOptimizer = None,
            opt2  : AbstractOptimizer = None
            ):
        super().__init__(space)
        self.eta  = eta
        self.opt1 = opt1
        self.opt2 = opt2


        if self.eta is None:
            self.eta = np.inf # prior optimu

        if self.opt1 is None:
            # NOTE: optimizer for stage one, vallina BO
            self.opt1 = HEBO(space)

        if  self.opt2 is None:
            # NOTE: optimizer for stage two, focus more on exploitation
            self.opt2 = BO(space, acq_conf = {'kappa' : 0.6})  


    def observe(self, x : pd.DataFrame, y : np.ndarray):
        self.opt1.observe(x, y)
        self.opt2.observe(x, y)

    def suggest(self, n_suggestions = 1, fix_input : dict = None):
        assert n_suggestions == 1
        if self.opt1.y is None or self.opt1.y.shape[0] == 0 or self.opt1.y.min() > self.eta:
            return self.opt1.suggest(n_suggestions, fix_input)
        return self.opt2.suggest(n_suggestions, fix_input)

    @property
    def best_x(self) -> pd.DataFrame:
        return self.opt1.best_x if self.opt1.best_y < self.opt2.best_y else self.opt2.best_x

    @property
    def best_y(self) -> float:
        return self.opt1.best_y if self.opt1.best_y < self.opt2.best_y else self.opt2.best_y
