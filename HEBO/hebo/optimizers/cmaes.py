# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch.distributions import MultivariateNormal

from hebo.design_space import DesignSpace
from .abstract_optimizer import AbstractOptimizer

class CMAES(AbstractOptimizer):
    support_parallel_opt  = True
    support_combinatorial = True
    def __init__(self, space : DesignSpace, pop_size = None, child_size = None, **algo_conf):
        super().__init__(space)
        self.dim           = self.space.num_paras
        self.algo_conf     = algo_conf
        self.pop_size      = pop_size
        self.child_size    = child_size
        self.restart_thres = algo_conf.get('restart_thres', 1e-4)
        if self.child_size is None:
            self.child_size = int(4 + np.floor(3 * np.log(self.dim)))
        if self.pop_size is None:
            self.pop_size = int(self.child_size / 2)
        assert self.pop_size < self.child_size, "Population size should be smaller than children size"

        self.weights  = torch.FloatTensor(np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size + 1)))
        self.weights /= self.weights.sum()
        self.mu_eff   = 1.0 / (self.weights**2).sum()

        self.cc   = algo_conf.get('cc', (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim))
        self.cs   = algo_conf.get('cs', (self.mu_eff + 2) / (self.dim + self.mu_eff + 5))
        self.c_r1 = algo_conf.get('c_r1', 2. / ((self.dim + 1.3)**2 + self.mu_eff))
        self.c_mu = algo_conf.get('c_mu', min(1 - self.c_r1, 2 * (self.mu_eff - 2 + 1. / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff)))
        self.damp = algo_conf.get('damp', 1. + 2 * max(0., np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1.))

        self.n_eval        = 0.
        self._best_x       = None
        self._best_y       = np.inf
        self.mu            = self.init_mu()
        self.sigma         = self.init_sigma()
        self.C             = self.init_C()
        self.p_sigma       = torch.zeros(self.dim)
        self.p_c           = torch.zeros(self.dim)
        self.n_resample    = 10

        self.norm_rand     = np.sqrt(self.dim) * (1 - 1. / (4 * self.dim) + 1. / (21 * self.dim**2))

        self.px            = None # parents
        self.cx            = None # children

    @property
    def max_step(self):
        return self.sigma * self.C.diag().max()

    def init_mu(self):
        # mu = self.space.opt_lb + (self.space.opt_ub - self.space.opt_lb) * torch.rand(self.dim)
        # return mu
        return 0.5 * (self.space.opt_lb + self.space.opt_ub).view(-1)

    def init_C(self):
        return torch.eye(self.dim)

    def init_sigma(self):
        return torch.tensor(0.5)

    def restart(self, start_from_mu = False):
        if not start_from_mu: 
                self.mu = self.init_mu()
        self.sigma   = self.init_sigma()
        self.p_sigma = torch.zeros(self.dim)
        self.C       = self.init_C()
        self.p_c     = torch.zeros(self.dim)

    def suggest(self, n_suggestions = None, fix_input : dict = None):
        """
        Perform optimisation and give recommendation using data observed so far
        ---------------------
        n_suggestions:  number of recommendations in this iteration

        fix_input:      parameters NOT to be optimized, but rather fixed, this
                        can be used for contextual BO, where you can set the contex as a design
                        parameter and fix it during optimisation
        """
        assert n_suggestions is None or n_suggestions == self.child_size
        assert fix_input is None

        n_suggestions = self.child_size
        space         = self.space
        lb            = space.opt_lb.view(-1).float()
        ub            = space.opt_ub.view(-1).float()

        if self.max_step < self.restart_thres * (ub - lb).norm():
            self.restart(np.random.choice([True, False]))

        try:
            dist = MultivariateNormal(self.mu, self.sigma * self.C)
        except (ValueError, RuntimeError):
            warnings.warn('failed to construct gaussian, restart')
            self.restart(True)
            dist = MultivariateNormal(self.mu, self.sigma * self.C)
        sample    = dist.sample((n_suggestions, ))
        sample[0] = self.mu
        
        # Handling bound constraints, resample then reflection
        for i in range(self.n_resample):
            cond = (sample >= lb).all() and (sample <= ub).all()
            if cond:
                break
            else:
                sample = torch.cat([sample, dist.sample((n_suggestions, ))], dim = 0)
                sample = sample[(sample >= lb).all(dim = 1) & (sample <= ub).all(dim = 1)]
                sample = sample[:n_suggestions]
        
        if sample.shape[0] < n_suggestions:
            sample = torch.cat([sample, dist.sample((n_suggestions - sample.shape[0], ))], dim = 0)
        
        while (sample < lb).any() or (sample > ub).any():
            for i in range(self.dim):
                rl    = 2 * lb[i] - sample[:, i]
                ru    = 2 * ub[i] - sample[:, i]
                vio_l = sample[:, i] < lb[i]
                vio_u = sample[:, i] > ub[i]
                sample[vio_l, i] = rl[vio_l]
                sample[vio_u, i] = ru[vio_u]

        X       = sample[:, :self.space.num_numeric]
        Xe      = sample[:, self.space.num_numeric:].round().long()
        self.cx = self.space.inverse_transform(X, Xe).head(n_suggestions)
        return self.cx

    def observe(self, x : pd.DataFrame, y : np.ndarray):
        """
        Observe new data
        """
        if self.cx is not None:
            assert (self.cx.values == x[self.cx.columns].values).all()
       
        if y.min() < self._best_y:
            self._best_y = y.min()
            self._best_x = x.iloc[[y.reshape(-1).argmin()]]

        self.n_eval += y.shape[0]

        y       = y.reshape(-1)
        lb      = self.space.opt_lb.view(-1).float()
        ub      = self.space.opt_ub.view(-1).float()
        cx, cxe = self.space.transform(x)
        cx      = torch.cat([cx, cxe.float()], dim = 1)

        self.px = x.iloc[y.argsort()[:self.pop_size]].copy()
        px, pxe = self.space.transform(self.px)
        px      = torch.cat([px, pxe.float()], dim = 1)

        mu_old        = self.mu.clone()
        self.mu       = (px.t() * self.weights).sum(axis = 1)

        D, P          = self.C.eig(True)
        D             = D[:, 0] * torch.eye(self.dim)
        Dinv          = (1. / D.diag()) * torch.eye(self.dim)
        C_half_inv    = P.mm(Dinv.sqrt()).mm(P.t())

        self.p_sigma  = (1 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * C_half_inv.mm((self.mu - mu_old).view(-1, 1) / self.sigma).view(self.p_sigma.shape)

        h_sig = 0.
        gen   = self.n_eval / self.child_size
        if self.p_sigma.norm() / np.sqrt(1 - (1 - self.cs)**(2 * gen + 1)) < (1.4 + 2 / (self.dim + 1)) * self.norm_rand:
            h_sig = 1.0
        self.p_c = (1 - self.cc)  * self.p_c + h_sig * np.sqrt(self.cc  * (2 - self.cc)   * self.mu_eff) * ((self.mu - mu_old) / self.sigma)

        self.sigma *= np.exp((self.cs / self.damp) * (self.p_sigma.norm() / self.norm_rand - 1) )
        if not torch.isfinite(self.sigma):
            self.sigma   = self.init_sigma()
            self.p_sigma = torch.zeros(self.dim)
        self.sigma = torch.min(2 * (ub - lb).norm(), self.sigma)

        # rank-mu estimation
        C_mu = torch.zeros(self.dim, self.dim)
        for i in range(self.pop_size):
            y     = (px[i] - mu_old).view(-1, 1) / self.sigma
            C_mu += self.weights[i] * y.mm(y.t())
        
        # rank-1 estimation
        C_r1 = self.p_c.view(-1, 1).mm(self.p_c.view(1, -1)) + (1 - h_sig) * self.cc * (2 - self.cc) * self.C

        self.C = (1 - self.c_r1 - self.c_mu) * self.C + self.c_r1 * C_r1 + self.c_mu * C_mu

    @property
    def best_x(self) -> pd.DataFrame:
        if self._best_x is None:
            raise RuntimeError('No data has been observed!')
        return self._best_x

    @property
    def best_y(self) -> pd.DataFrame:
        if self.best_x is None:
            raise RuntimeError('No data has been observed!')
        return self._best_y
