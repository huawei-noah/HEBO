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
from hebo.models.model_factory import get_model, model_dict
from hebo.acquisitions.acq import GeneralAcq
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

from .abstract_optimizer import AbstractOptimizer

class GeneralBO(AbstractOptimizer):
    """
    Bayesian optimisation that supports multi-objective and constrained optimization
    """
    def __init__(self,
            space : DesignSpace,
            num_obj:     int   = 1,
            num_constr:  int   = 0,
            rand_sample: int   = None,
            model_name:  str   = 'deep_ensemble',
            model_conf:  dict  = None,
            kappa:       float = 2.,
            c_kappa:     float = 0.,
            use_noise:   bool  = False
            ):
        super().__init__(space)
        self.space       = space
        self.num_obj     = num_obj
        self.num_constr  = num_constr
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else rand_sample
        self.model_name  = model_name
        self.model_conf  = model_conf if model_conf is not None else {}
        self.X           = pd.DataFrame(columns = self.space.para_names)
        self.y           = np.zeros((0, num_obj + num_constr))
        self.kappa       = kappa
        self.c_kappa     = c_kappa
        self.use_noise   = use_noise
        self.model       = None
        self.evo_pop     = 100
        self.evo_iters   = 200
        assert model_dict.get(model_name) is not None
        if num_obj + num_constr > 1:
            assert model_dict.get(model_name).support_multi_output

    def suggest(self, n_suggestions = 1, fix_input = None):
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
            self.model = get_model(self.model_name, X.shape[1], Xe.shape[1], y.shape[1], num_uniqs = num_uniqs, **self.model_conf)
            self.model.fit(X, Xe, y)

            acq = GeneralAcq(
                  self.model,
                  self.num_obj,
                  self.num_constr,
                  kappa = self.kappa,
                  c_kappa = self.c_kappa,
                  use_noise = self.use_noise)
            opt     = EvolutionOpt(self.space, acq, pop = self.evo_pop, iters = self.evo_iters)
            suggest = opt.optimize()
            with torch.no_grad():
                py, ps2 = self.model.predict(*self.space.transform(suggest))
                largest_uncert_id = np.argmax(np.log(ps2).sum(axis = 1))
            if suggest.shape[0] >= n_suggestions:
                selected_id = np.random.choice(suggest.shape[0], n_suggestions).tolist()
                if largest_uncert_id not in selected_id:
                    selected_id[0] = largest_uncert_id
                return suggest.iloc[selected_id]
            else:
                rand_samp = self.space.sample(n_suggestions - suggest.shape[0])
                suggest   = pd.concat([suggest, rand_samp], axis = 0, ignore_index = True)
                return suggest

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

    def get_pf(self, y : np.ndarray) -> pd.DataFrame:
        pass

    @property
    def best_x(self):
        raise NotImplementedError('Not implemented for multi-objective algorithm')

    @property
    def best_y(self):
        raise NotImplementedError('Not implemented for multi-objective algorithm')
