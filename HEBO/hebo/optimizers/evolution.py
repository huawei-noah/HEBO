# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

"""
Evolutionary optimzation, pymoo wrapper with HEBO API
"""

import numpy as np
import pandas as pd
import torch

# from pymoo.factory import get_mutation, get_crossover, get_algorithm
# from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.core.problem import Problem
from pymoo.config import Config
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
Config.show_compile_hint = False

from hebo.design_space.design_space import DesignSpace
from .abstract_optimizer import AbstractOptimizer
from hebo.acq_optimizers.evolution_optimizer import space_to_pymoo_vars

class DummyProb(Problem):
    def __init__(self,
            space      : DesignSpace, 
            num_obj    : int,
            num_constr : int
            ):
        vars = space_to_pymoo_vars(space)
        super().__init__(vars = vars, n_obj = num_obj, n_constr = num_constr)
        self.space      = space
        self.num_obj    = num_obj
        self.num_constr = num_constr

    def _evaluate(self, x, out : dict, *args, **kwargs):
        pass

class Evolution(AbstractOptimizer):
    support_parallel_opt    = True
    support_constraint      = True
    support_multi_objective = True
    support_combinatorial   = True

    def __init__(self, 
            space      : DesignSpace,
            num_obj    : int  = 1,
            num_constr : int  = 0, # NOTE: single-objective unconstrained optimization by default
            algo       : str  = 'nsga2',
            verbose    : bool = False,
            **algo_conf
            ):
        super().__init__(space)
        self.num_obj    = num_obj
        self.num_constr = num_constr
        if algo == 'ga':
            self.algo = MixedVariableGA(**algo_conf)
        elif algo == 'nsga2':
            self.algo = NSGA2(
                    sampling = MixedVariableSampling(), 
                    mating   = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()), 
                    eliminate_duplicates = MixedVariableDuplicateElimination(), 
                    **algo_conf, 
                    )
        else:
            raise ValueError(f'Only ga and nsga2 supported, unrecognized algorithm {algo}')
        self.prob = DummyProb(self.space, self.num_obj, self.num_constr)
        self.algo.setup(self.prob, termination = ('n_gen', np.inf), verbose = verbose)
        self.n_observation = 0

    def suggest(self, n_suggestions = None):
        self.pop = self.algo.ask()
        pop_x    = self.pop.get('X')
        x        = []
        xe       = []
        for n in self.space.numeric_names:
            x.append(np.vectorize(lambda x : x[n])(pop_x))
        for n in self.space.enum_names:
            xe.append(np.vectorize(lambda x : x[n])(pop_x))
        x   = torch.FloatTensor(x).t()
        xe  = torch.LongTensor(xe).t()
        rec = self.space.inverse_transform(x, xe)
        return rec

    def observe(self, rec : pd.DataFrame, obs : np.ndarray):
        x, xe = self.space.transform(rec)
        x_cat = torch.cat([x, xe.float()], dim = 1).numpy()
        obj   = obs[:, :self.num_obj]
        vio   = obs[:, self.num_obj:]
        if self.num_constr > 0:
            self.pop.set('F', obj)
            self.pop.set('G', vio)
        else:
            self.pop.set('F', obj)
        self.algo.tell(infills = self.pop)
        self.n_observation += rec.shape[0]

    @property
    def best_x(self) -> pd.DataFrame:
        if self.n_observation == 0:
            raise RuntimeError('No data has been observed')
        pop_x = self.algo.opt.get('X')
        x     = []
        xe    = []
        for n in self.space.numeric_names:
            x.append(np.vectorize(lambda x : x[n])(pop_x))
        for n in self.space.enum_names:
            xe.append(np.vectorize(lambda x : x[n])(pop_x))
        x   = torch.FloatTensor(x).t()
        xe  = torch.LongTensor(xe).t()
        rec = self.space.inverse_transform(x, xe)
        return rec

    @property
    def best_y(self) -> np.ndarray:
        if self.n_observation == 0:
            raise RuntimeError('No data has been observed')
        opt    = self.algo.opt
        best_y = opt.get('F')
        if self.num_constr > 0:
            vio = opt.get('G')
            best_y = np.hstack([best_y, vio])
        return best_y
