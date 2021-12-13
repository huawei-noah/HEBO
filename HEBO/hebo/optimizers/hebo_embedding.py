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
from sklearn.preprocessing import power_transform

from hebo.design_space.design_space import DesignSpace
from hebo.design_space.numeric_param import NumericPara
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import Acquisition, MACE, Mean, Sigma
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

from .abstract_optimizer import AbstractOptimizer
from .hebo import HEBO

torch.set_num_threads(min(1, torch.get_num_threads()))

def gen_emb_space(eff_dim : int, scale : float) -> DesignSpace:
    scale = -1 * scale if scale < 0 else scale
    space = DesignSpace().parse([{'name' : f'y{i}', 'type' : 'num', 'lb' : -1 * scale, 'ub' : scale} for i in range(eff_dim)])
    return space

def check_design_space(space : DesignSpace) -> bool:
    """
    All parameters should be continuous parameters and the range should be [-1, 1]
    """
    for k, v in space.paras.items():
        if not isinstance(v, NumericPara):
            return False
    lb = space.opt_lb
    ub = space.opt_ub
    if not (lb + torch.ones(space.num_paras)).abs().sum() < 1e-6:
        return False
    if not (ub - torch.ones(space.num_paras)).abs().sum() < 1e-6:
        return False
    return True

def gen_proj_matrix(eff_dim : int, dim : int, strategy : str = 'alebo'):
    if strategy == 'hesbo':
        matrix = np.zeros((eff_dim,dim))
        for i in range(dim):
            sig = np.random.choice([-1,1])
            idx = np.random.choice(eff_dim)
            matrix[idx,i] = sig * 1.0
    else:
        matrix =  np.random.randn(eff_dim, dim)
        if strategy == 'alebo':
            matrix = matrix / np.sqrt((matrix**2).sum(axis = 0))
    return matrix

def gen_mace_cls(proj_matrix):
    class MACE_Embedding(Acquisition):
        def __init__(self, model, best_y, **conf):
            super().__init__(model, **conf)
            self.mace        = MACE(model, best_y, **conf)
            self.proj_matrix = torch.FloatTensor(proj_matrix)
        @property
        def num_constr(self):
            return 1

        @property
        def num_obj(self):
            return 3

        def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
            assert xe is None or xe.shape[1] == 0
            mace_acq  = self.mace(x, xe)
            x_orig    = torch.mm(x, self.proj_matrix)
            bound_vio = (x_orig.abs() - 1.0).clamp(min = 0.).sum(axis = 1).view(-1, 1)
            return torch.cat([mace_acq, bound_vio], dim = 1)
    return MACE_Embedding

class HEBO_Embedding(AbstractOptimizer):
    support_parallel_opt  = True
    support_combinatorial = False
    support_contextual    = False
    def __init__(self,
            space : DesignSpace, 
            model_name      = 'gpy',
            eff_dim : int   = 1,
            scale   : float = 1,
            strategy : str  = 'alebo',
            clip : bool     = False,
            rand_sample     = None):
        super().__init__(space)
        assert check_design_space(space)
        self.space       = space
        self.scale       = scale
        self.eff_dim     = eff_dim
        self.proj_matrix = gen_proj_matrix(eff_dim, space.num_paras, strategy)
        self.eff_space   = gen_emb_space(eff_dim, scale)
        self.clip        = clip
        self.acq_cls     = MACE if self.clip else gen_mace_cls(self.proj_matrix) # If we use
        self.mace        = HEBO(self.eff_space, model_name, rand_sample, acq_cls = self.acq_cls)
        self.mace.quasi_sample = self.quasi_sample

    def quasi_sample(self, n, fix_input = None, factor = 16): 
        assert fix_input is None
        if self.clip:
            return self.eff_space.sample(n)

        B    = torch.FloatTensor(self.proj_matrix)
        L    = torch.cholesky(B.mm(B.t()))
        samp = pd.DataFrame(columns = self.eff_space.numeric_names)
        while samp.shape[0] < n:
            samp_hd = self.space.sample(100)
            alpha   = B.mm(torch.FloatTensor(samp_hd.values.T))
            samp_ld = pd.DataFrame(factor * torch.cholesky_solve(alpha, L).t().numpy(), columns = samp.columns)
            samp_pj = self.project(samp_ld)
            samp_ld = samp_ld[samp_pj.max(axis = 1) <= 1.0]
            samp_ld = samp_ld[samp_pj.min(axis = 1) >= -1.0]
            if samp_ld.shape[0] == samp_hd.shape[0]:
                factor /= 0.8
                continue
            elif samp_ld.shape[0] == 0:
                factor *= 0.8
            samp = samp.append(samp_ld, ignore_index = True)
        return samp.head(n)

    def project(self, df_x_ld : pd.DataFrame) -> pd.DataFrame:
        x    = df_x_ld[self.eff_space.numeric_names].values
        x_hd = np.matmul(x, self.proj_matrix)
        if self.clip:
            x_hd = np.tanh(x_hd)
        return pd.DataFrame(x_hd, columns = self.space.numeric_names)

    def suggest(self, n_suggestions : int = 1):
        df_suggest = self.mace.suggest(n_suggestions)
        return df_suggest 

    def observe(self, X, y):
        self.mace.observe(X, y)

    @property
    def best_x(self)->pd.DataFrame:
        return self.mace.best_x

    @property
    def best_y(self)->float:
        return self.mace.best_y
