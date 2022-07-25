# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy  as np
import pandas as pd
import torch

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model, get_model_class
from hebo.acquisitions.acq import GeneralAcq
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

from .abstract_optimizer import AbstractOptimizer

from pymoo.factory import get_performance_indicator
from pymoo.util.dominator import Dominator

class GeneralBO(AbstractOptimizer):
    """
    Bayesian optimisation that supports multi-objective and constrained optimization
    """
    def __init__(self,
            space : DesignSpace,
            num_obj:      int   = 1,
            num_constr:   int   = 0,
            rand_sample:  int   = None,
            model_name:   str   = 'multi_task',
            model_config: dict  = None,
            kappa:        float = 2.,
            c_kappa:      float = 0.,
            use_noise:    bool  = False, 
            evo_pop:      int   = 100,
            evo_iters:    int   = 200,
            ref_point:    np.ndarray = None,
            ):
        super().__init__(space)
        self.space        = space
        self.num_obj      = num_obj
        self.num_constr   = num_constr
        self.rand_sample  = 1 + self.space.num_paras if rand_sample is None else rand_sample
        self.model_name   = model_name
        self.model_config = model_config if model_config is not None else {}
        self.X            = pd.DataFrame(columns = self.space.para_names)
        self.y            = np.zeros((0, num_obj + num_constr))
        self.kappa        = kappa
        self.c_kappa      = c_kappa
        self.use_noise    = use_noise
        self.model        = None
        self.evo_pop      = evo_pop
        self.evo_iters    = evo_iters
        self.iter         = 0
        self.ref_point    = ref_point
        if num_obj + num_constr > 1:
            assert get_model_class(model_name).support_multi_output

    def suggest(self, n_suggestions = 1, fix_input = None):
        self.iter += 1
        if self.X.shape[0] < self.rand_sample:
            sample = self.space.sample(n_suggestions)
            if fix_input is not None:
                for k, v in fix_input.items():
                    sample[k] = v
            return sample
        else:
            X, Xe      = self.space.transform(self.X)
            y          = torch.FloatTensor(self.y)
            num_uniqs  = None if Xe.shape[1] == 0 else [len(self.space.paras[name].categories) for name in self.space.enum_names]
            self.model = get_model(self.model_name, X.shape[1], Xe.shape[1], y.shape[1], num_uniqs = num_uniqs, **self.model_config)
            self.model.fit(X, Xe, y)
            kappa   = self.kappa
            c_kappa = self.c_kappa
            upsi    = 0.1
            delta   = 0.01
            if kappa is None:
                kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(self.iter) + np.log(3 * np.pi**2 / (3 * delta))))
            if c_kappa is None:
                c_kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(self.iter) + np.log(3 * np.pi**2 / (3 * delta))))
            acq = GeneralAcq(
                  self.model,
                  self.num_obj,
                  self.num_constr,
                  kappa     = kappa, 
                  c_kappa   = c_kappa,
                  use_noise = self.use_noise)
            opt     = EvolutionOpt(self.space, acq, pop = self.evo_pop, iters = self.evo_iters)
            suggest = opt.optimize()
            if suggest.shape[0] < n_suggestions:
                rand_samp = self.space.sample(n_suggestions - suggest.shape[0])
                suggest   = pd.concat([suggest, rand_samp], axis = 0, ignore_index = True)
                return suggest
            elif self.ref_point is None:
                with torch.no_grad():
                    py, ps2 = self.model.predict(*self.space.transform(suggest))
                    largest_uncert_id = np.argmax(np.log(ps2).sum(axis = 1))
                select_id = np.random.choice(suggest.shape[0], n_suggestions, replace = False).tolist()
                if largest_uncert_id not in select_id:
                    select_id[0] = largest_uncert_id
                return suggest.iloc[select_id]
            else:
                assert self.num_obj > 1
                assert self.num_constr == 0
                n_mc = 10
                hv   = get_performance_indicator('hv', ref_point = self.ref_point.reshape(-1))
                with torch.no_grad():
                    py, ps2 = self.model.predict(*self.space.transform(suggest))
                    y_samp  = self.model.sample_y(*self.space.transform(suggest), n_mc).numpy()
                y_curr    = self.get_pf(self.y).copy()
                select_id = []
                for i in range(n_suggestions):
                    ehvi_lst = []
                    base_hv  = hv.do(y_curr)
                    for j in range(suggest.shape[0]):
                        samp    = y_samp[:, j]
                        hvi_est = 0
                        for k in range(n_mc):
                            y_tmp    = np.vstack([y_curr, samp[[k]]])
                            hvi_est += hv.do(y_tmp) - base_hv
                        hvi_est /= n_mc
                        ehvi_lst.append(hvi_est)
                    best_id = np.argmax(ehvi_lst) if max(ehvi_lst) > 0 else np.random.choice(suggest.shape[0])
                    y_curr  = np.vstack([y_curr, y_samp[:, best_id].min(axis = 0, keepdims = True)])
                    select_id.append(best_id)

            select_id = list(set(select_id))
            if len(select_id) < n_suggestions:
                candidate_id = [i for i in range(suggest.shape[0]) if i not in select_id]
                select_id   += np.random.choice(candidate_id, n_suggestions - len(select_id), replace = False).tolist()
            return suggest.iloc[select_id]

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        valid_id = np.isfinite(y).all(axis = 1)
        XX       = X.iloc[valid_id]
        yy       = y[valid_id]
        self.X   = self.X.append(XX, ignore_index = True)
        self.y   = np.vstack([self.y, yy])
        assert self.y.shape[1] == self.num_obj + self.num_constr

    def select_best(self, rec : pd.DataFrame) -> pd.DataFrame:
        pass

    def get_pf(self, y : np.ndarray, return_optimal = False) -> pd.DataFrame:
        feasible = (y[:, self.num_obj:] <= 0).all(axis = 1)
        y        = y[feasible].copy()
        y_obj    = y[:, :self.num_obj]
        if feasible.any():
            dom_mat = Dominator().calc_domination_matrix(y_obj, None)
            optimal = (dom_mat >= 0).all(axis = 1)
        else:
            optimal = feasible

        if not return_optimal:
            return y[optimal].copy()
        else:
            return optimal


    @property
    def best_x(self):
        optimal = self.get_pf(self.y, return_optimal = True)
        return self.X[optimal].copy()

    @property
    def best_y(self):
        return self.get_pf(self.y)
