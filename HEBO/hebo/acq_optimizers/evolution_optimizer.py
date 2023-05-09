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
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.config import Config
Config.show_compile_hint = False

from ..design_space.design_space import DesignSpace
from ..acquisitions.acq import Acquisition

def space_to_pymoo_vars(space):
    vars = {}
    for i, p_name in enumerate(space.para_names):
        p  = space.paras[p_name]
        lb = space.opt_lb[i].item()
        ub = space.opt_ub[i].item()
        if p.is_numeric:
            if not p.is_discrete_after_transform:
                vars[p_name] = Real(bounds = (lb, ub))
            else:
                vars[p_name] = Integer(bounds = (lb, ub))
        elif p.is_categorical:
            vars[p_name] = Choice(options = list(range(int(lb), int(ub + 1))))
        else:
            raise ValueError('Un recognoized parameter type {p_name}')
    return vars

def get_init_pop(space, pop, initial_suggest : pd.DataFrame = None, sobol_init = False) -> Population:
    if sobol_init:
        init_pop = space.sample(pop)
    else:
        eng        = SobolEngine(space.num_paras, scramble = True)
        sobol_samp = eng.draw(pop)
        sobol_samp = sobol_samp * (space.opt_ub - space.opt_lb) + space.opt_lb
        x          = sobol_samp[:, :space.num_numeric]
        xe         = sobol_samp[:, space.num_numeric:].round().long()
        for i, n in enumerate(space.numeric_names):
            if space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        init_pop = space.inverse_transform(x, xe)
    if initial_suggest is not None:
        init_pop = pd.concat([initial_suggest, init_pop], axis = 0).head(pop)
    x, xe    = space.transform(init_pop)
    init_pop = np.hstack([x.numpy(), xe.numpy().astype(float)])
    pop_lst  = []
    for item in init_pop:
        pop_item = {}
        for i, name in enumerate(space.para_names):
            if name in space.enum_names:
                pop_item[name] = int(item[i])
            else:
                pop_item[name] = item[i]
        pop_lst.append(pop_item)
    pop = Population.new(X = pop_lst)
    return pop

class BOProblem(Problem):
    def __init__(self,
            acq   : Acquisition,
            space : DesignSpace, 
            fix   : dict = None 
            ):
        self.acq   = acq
        self.space = space
        self.fix   = fix # NOTE: use self.fix to enable contextual BO
        vars       = space_to_pymoo_vars(self.space)
        super().__init__(vars = vars, n_obj = acq.num_obj, n_constr = acq.num_constr)

    def _evaluate(self, para : np.ndarray, out : dict, *args, **kwargs):
        ## TODO: Use the repair operator to handle fix_input
        num_x = para.shape[0]
        x  = []
        xe = []
        for n in self.space.numeric_names:
            func = np.vectorize(lambda x : x[n])
            x.append(func(para))
        for n in self.space.enum_names:
            func = np.vectorize(lambda x : x[n])
            xe.append(func(para))
        x  = torch.FloatTensor(x).t().reshape(num_x, -1)
        xe = torch.LongTensor(xe).t().reshape(num_x, -1)
        if self.fix is not None:
            df_x = self.space.inverse_transform(x, xe)
            for k, v in self.fix.items():
                df_x[k] = v
            x, xe = self.space.transform(df_x)
        with torch.no_grad():
            acq_v = self.acq(x, xe)
        out['F'] = acq_v[:, :self.acq.num_obj].numpy()
        out['G'] = acq_v[:, self.acq.num_obj:].numpy()

class EvolutionOpt:
    def __init__(self,
            design_space : DesignSpace,
            acq          : Acquisition,
            es           : str = None, 
            **conf):
        self.space      = design_space
        self.es         = es 
        self.acq        = acq
        self.pop        = conf.get('pop', 100)
        self.iter       = conf.get('iters',500)
        self.verbose    = conf.get('verbose', False)
        self.repair     = conf.get('repair', None)
        self.sobol_init = conf.get('sobol_init', True)
        assert(self.acq.num_obj > 0)

        if self.es is None:
            self.es = 'nsga2' if self.acq.num_obj > 1 else 'ga'


    def optimize(self, initial_suggest : pd.DataFrame = None, fix_input : dict = None, return_pop = False) -> pd.DataFrame:
        lb        = self.space.opt_lb.numpy()
        ub        = self.space.opt_ub.numpy()
        prob      = BOProblem(self.acq, self.space, fix_input)
        init_pop  = get_init_pop(self.space, self.pop, initial_suggest, self.sobol_init)
        if self.acq.num_obj == 1:
            algo = MixedVariableGA(pop_size = self.pop, repair = self.repair, sampling = init_pop)
        else:
            algo = NSGA2(pop_size = self.pop, 
                    sampling = init_pop, 
                    mating   = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()), 
                    eliminate_duplicates = MixedVariableDuplicateElimination()
                    )
        res = minimize(prob, algo, ('n_gen', self.iter), verbose = self.verbose)
        if res.X is not None and not return_pop:
            x = res.X
            if isinstance(x, dict):
                x = [x]
            if isinstance(x, np.ndarray):
                x = x.tolist()
            opt_x = pd.DataFrame(x)[self.space.para_names].values.astype(float)
        else:
            opt_x = pd.DataFrame([p.X for p in res.pop])[self.space.para_names].values.astype(float)
            if self.acq.num_obj == 1 and not return_pop:
                opt_x = opt_x[[np.random.choice(opt_x.shape[0])]]
        
        self.res  = res
        opt_xcont = torch.from_numpy(opt_x[:, :self.space.num_numeric])
        opt_xenum = torch.from_numpy(opt_x[:, self.space.num_numeric:])
        df_opt    = self.space.inverse_transform(opt_xcont, opt_xenum)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_opt[k] = v
        return df_opt
