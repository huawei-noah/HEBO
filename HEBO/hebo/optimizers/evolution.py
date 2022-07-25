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

from pymoo.factory import get_mutation, get_crossover, get_algorithm
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.core.problem import Problem
from pymoo.config import Config
Config.show_compile_hint = False

from hebo.design_space.design_space import DesignSpace
from .abstract_optimizer import AbstractOptimizer

class DummyProb(Problem):
    def __init__(self,
            lb         : np.ndarray,
            ub         : np.ndarray,
            num_obj    : int,
            num_constr : int
            ):
        super().__init__(len(lb), xl = lb, xu = ub, n_obj = num_obj, n_constr = num_constr)

    def _evaluate(self, x, out : dict, *args, **kwargs):
        for k, v in kwargs.items():
            out[k] = v

class Evolution(AbstractOptimizer):
    support_parallel_opt    = True
    support_constraint      = True
    support_multi_objective = True
    support_combinatorial   = True
    support_contextual      = False

    def __init__(self, 
            space      : DesignSpace,
            num_obj    : int  = 1,
            num_constr : int  = 0, # NOTE: single-objective unconstrained optimization by default
            algo       : str  = 'nsga2',
            verbose    : bool = False,
            **algo_conf
            ):
        super().__init__(space)
        if algo is None:
            algo = 'ga' if num_obj == 1 else 'nsga2'

        self.num_obj    = num_obj
        self.num_constr = num_constr
        if algo in ['ga', 'nsga2']:
            self.algo = get_algorithm(algo, mutation = self.get_mutation(), crossover = self.get_crossover(), **algo_conf)
        else:
            self.algo = get_algorithm(algo, **algo_conf)
        lb = self.space.opt_lb.numpy()
        ub = self.space.opt_ub.numpy()
        self.prob = DummyProb(lb, ub, self.num_obj, self.num_constr)
        self.algo.setup(self.prob, ('n_gen', np.inf), verbose = verbose)
        self.n_observation = 0

    def suggest(self, n_suggestions = None, fix_input : dict = None):
        self.pop = self.algo.ask()
        pop_x    = torch.from_numpy(self.pop.get('X').astype(float)).float()
        x        = pop_x[:, :self.space.num_numeric]
        xe       = pop_x[:, self.space.num_numeric:].round().long()
        rec      = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                rec[k] = v
        x, xe = self.space.transform(rec)
        x_cat = torch.cat([x, xe.float()], dim = 1).numpy()
        self.pop.set('X', x_cat)
        return rec

    def observe(self, rec : pd.DataFrame, obs : np.ndarray):
        x, xe = self.space.transform(rec)
        x_cat = torch.cat([x, xe.float()], dim = 1).numpy()
        obj   = obs[:, :self.num_obj]
        vio   = obs[:, self.num_obj:]

        self.pop.set('X', x_cat)
        if self.num_constr > 0:
            self.algo.evaluator.eval(self.prob, self.pop, F = obj, G = vio)
        else:
            self.algo.evaluator.eval(self.prob, self.pop, F = obj)
        self.algo.tell(infills = self.pop)
        self.n_observation += rec.shape[0]

    @property
    def best_x(self) -> pd.DataFrame:
        if self.n_observation == 0:
            raise RuntimeError('No data has been observed')
        opt = torch.from_numpy(self.algo.opt.get('X')).float()
        x   = opt[:, :self.space.num_numeric]
        xe  = opt[:, self.space.num_numeric:].round().long()
        return self.space.inverse_transform(x, xe)

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

    def get_mutation(self):
        mask = []
        for name in (self.space.numeric_names + self.space.enum_names):
            if self.space.paras[name].is_discrete_after_transform:
                mask.append('int')
            else:
                mask.append('real')

        mutation = MixedVariableMutation(mask, {
            'real' : get_mutation('real_pm', eta = 20), 
            'int'  : get_mutation('int_pm', eta = 20)
        })
        return mutation

    def get_crossover(self):
        mask = []
        for name in (self.space.numeric_names + self.space.enum_names):
            if self.space.paras[name].is_discrete_after_transform:
                mask.append('int')
            else:
                mask.append('real')

        crossover = MixedVariableCrossover(mask, {
            'real' : get_crossover('real_sbx', eta = 15, prob = 0.9), 
            'int'  : get_crossover('int_sbx', eta = 15, prob = 0.9)
        })
        return crossover
