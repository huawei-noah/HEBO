# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import sys
import numpy  as np
import pandas as pd
import torch
from copy import deepcopy
from torch.quasirandom import SobolEngine

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import NoisyAcq, Mean, Sigma
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
from .hebo import HEBO

torch.set_num_threads(min(1, torch.get_num_threads()))

class NoisyOpt(HEBO):
    support_parallel_opt  = True
    support_combinatorial = True
    support_contextual    = True
    def __init__(self, space, model_name = 'gpy', rand_sample = None, es = 'nsga2', model_config = None):
        """
        model_name : surrogate model to be used
        rand_iter  : iterations to perform random sampling
        """
        super().__init__(space, model_name, rand_sample, NoisyAcq, es, model_config)

    def suggest(self, n_suggestions=1, fix_input = None):
        assert fix_input is None
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            y     = torch.FloatTensor(self.y).clone()
            model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
            model.fit(X, Xe, y)

            best_id = self.get_best_id(fix_input)
            best_x  = self.X.iloc[[best_id]]
            best_y  = y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()


            acq = self.acq_cls(model, 1, 0)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
            rec = opt.optimize(initial_suggest = best_x, return_pop = True)
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = rec.append(rand_rec, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break 
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec      = rec.append(rand_rec, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            x_guess   = []
            mu  = Mean(model)
            sig = Sigma(model, linear_a = -1.)
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            return rec_selected
