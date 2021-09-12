# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy  as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_mutation, get_crossover
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.optimize import minimize
from pymoo.model.problem import get_problem_from_func
from pymoo.configuration import Configuration
from ..design_space.design_space import DesignSpace
from ..acquisitions.acq import Acquisition
Configuration.show_compile_hint = False

class EvolutionOpt:
    def __init__(self,
            design_space : DesignSpace,
            acq          : Acquisition,
            **conf):
        self.space      = design_space
        self.acq        = acq
        self.pop        = conf.get('pop', 100)
        self.iter       = conf.get('iters',500)
        self.verbose    = conf.get('verbose', False)
        self.num_obj    = self.acq.num_obj
        self.num_constr = self.acq.num_constr
        assert(self.num_obj > 0)

    def callback(self):
        pass

    def pymoo_obj(self, x : np.array, out : dict, *args, **kwargs):
        num_x = x.shape[0]
        xcont = torch.from_numpy(x[:, :self.space.num_numeric].astype(float))
        xenum = torch.from_numpy(x[:, self.space.num_numeric:].astype(float))
        df_x  = self.space.inverse_transform(xcont, xenum)
        if self.fix is not None: # invalidate fixed input, replace with fixed values
            for k, v in self.fix.items():
                df_x[k] = v
        xcont, xenum = self.space.transform(df_x)

        with torch.no_grad():
            acq_eval = self.acq(xcont, xenum).numpy().reshape(num_x, self.num_obj + self.num_constr)
            out['F'] = acq_eval[:, :self.num_obj]

            if self.num_constr > 0:
                out['G'] = acq_eval[:, -1 * self.num_constr:]

        self.callback()

    def get_init_pop(self, initial_suggest : pd.DataFrame = None) -> np.ndarray:
        # init_pop = self.space.sample(self.pop)
        self.eng   = SobolEngine(self.space.num_paras, scramble = True)
        sobol_samp = self.eng.draw(self.pop)
        sobol_samp = sobol_samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x          = sobol_samp[:, :self.space.num_numeric]
        xe         = sobol_samp[:, self.space.num_numeric:]
        init_pop   = self.space.inverse_transform(x, xe)
        if initial_suggest is not None:
            init_pop = pd.concat([initial_suggest, init_pop], axis = 0).head(self.pop)
        x, xe = self.space.transform(init_pop)
        return np.hstack([x.numpy(), xe.numpy().astype(float)])

    def get_mutation(self):
        mask = []
        for name in (self.space.numeric_names + self.space.enum_names):
            if self.space.paras[name].is_discrete:
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
            if self.space.paras[name].is_discrete:
                mask.append('int')
            else:
                mask.append('real')

        crossover = MixedVariableCrossover(mask, {
            'real' : get_crossover('real_sbx', eta = 15, prob = 0.9), 
            'int'  : get_crossover('int_sbx', eta = 15, prob = 0.9)
        })
        return crossover

    def optimize(self, initial_suggest : pd.DataFrame = None, fix_input : dict = None) -> pd.DataFrame:
        self.fix  = fix_input
        lb        = self.space.opt_lb.numpy()
        ub        = self.space.opt_ub.numpy()
        prob      = get_problem_from_func(self.pymoo_obj, xl = lb,  xu = ub, n_var = len(lb))
        init_pop  = self.get_init_pop(initial_suggest)
        mutation  = self.get_mutation()
        crossover = self.get_crossover()
        if self.num_obj > 1:
            algo = NSGA2(pop_size = self.pop, sampling = init_pop, mutation = mutation, crossover = crossover)
        else:
            algo = GA(pop_size = self.pop, sampling = init_pop, mutation = mutation, crossover = crossover)
        res = minimize(prob, algo, ('n_gen', self.iter), verbose = self.verbose)
        if res.X is None: # no feasible solution founc
            opt_x = np.array([ind.X for ind in res.pop]).astype(float)
        else:
            opt_x = res.X.reshape(-1, len(lb)).astype(float)
        
        opt_xcont = torch.from_numpy(opt_x[:, :self.space.num_numeric])
        opt_xenum = torch.from_numpy(opt_x[:, self.space.num_numeric:])
        df_opt    = self.space.inverse_transform(opt_xcont, opt_xenum)
        if self.fix is not None:
            for k, v in self.fix.items():
                df_opt[k] = v
        return df_opt
