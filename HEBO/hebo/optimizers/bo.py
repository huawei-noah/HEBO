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
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import LCB
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

from .abstract_optimizer import AbstractOptimizer

class BO(AbstractOptimizer):
    support_combinatorial = True
    support_contextual    = True
    def __init__(
            self,
            space : DesignSpace,
            model_name  = 'gpy',
            rand_sample = None,
            acq_cls     = None, 
            acq_conf    = None):
        super().__init__(space)
        self.space       = space
        self.X           = pd.DataFrame(columns = self.space.para_names)
        self.y           = np.zeros((0, 1))
        self.model_name  = model_name
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        self.acq_cls     = LCB if acq_cls is None else acq_cls
        self.acq_conf    = {'kappa' : 2.0} if acq_conf is None else acq_conf

    def suggest(self, n_suggestions = 1, fix_input = None):
        assert n_suggestions == 1
        if self.X.shape[0] < self.rand_sample:
            sample = self.space.sample(n_suggestions)
            if fix_input is not None:
                for k, v in fix_input.items():
                    sample[k] = v
            return sample
        else:
            X, Xe     = self.space.transform(self.X)
            y         = torch.FloatTensor(self.y)
            num_uniqs = None if Xe.shape[1] == 0 else [len(self.space.paras[name].categories) for name in self.space.enum_names]
            model     = get_model(self.model_name, X.shape[1], Xe.shape[1], y.shape[1], num_uniqs = num_uniqs, warp = False)
            model.fit(X, Xe, y)

            acq = self.acq_cls(model, **self.acq_conf)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100)

            suggest = self.X.iloc[[np.argmin(self.y.reshape(-1))]]
            return opt.optimize(initial_suggest = suggest, fix_input = fix_input)

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
        assert(y.shape[1] == 1)
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX       = X.iloc[valid_id]
        yy       = y[valid_id].reshape(-1, 1)
        self.X   = self.X.append(XX, ignore_index = True)
        self.y   = np.vstack([self.y, yy])

    @property
    def best_x(self)->pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.X.iloc[[self.y.argmin()]]

    @property
    def best_y(self)->float:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.y.min()
