# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import pandas as pd
import numpy as np
import random
from scipy.spatial import KDTree

from pymoo.factory import get_sampling, get_crossover, get_mutation, get_algorithm
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import torch
import gpytorch

from hebo.design_space import DesignSpace
from hebo.models.model_factory import get_model, model_dict
from hebo.optimizers.abstract_optimizer import AbstractOptimizer


class MyProblem_gplocal(Problem):
    def __init__(self, gp_model, y_min, local_info, var_num=10, kappa=1.5, xi=1e-4,
                 lb=np.array([-1]*10), ub=np.array([1]*10), noise_level=0.0):
        self.gp_model = gp_model
        self.y_min = y_min
        self.local_info = local_info   # pos, radius, scale, nb_indices, tr_Z
        self.var_num = var_num
        self.kappa = kappa
        self.xi = xi
        self.lb = lb
        self.ub = ub
        self.noise_level = noise_level
        super().__init__(n_var=self.var_num, n_obj=1, n_constr=4, xl=self.lb, xu=self.ub, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        with torch.no_grad():
            mean, var = self.gp_model.predict(torch.FloatTensor(x.astype('float64')).reshape(1, -1), None)
            std = var.sqrt()
        out["F"] = mean.numpy() + self.noise_level*np.random.randn() - std.numpy()

        # in selected voronoi cell
        d_one = []
        for k in self.local_info[3]:
            ds = np.linalg.norm(x - self.local_info[4][self.local_info[0]])
            dss = np.linalg.norm(x - self.local_info[4][k])
            d_one.append(self.local_info[2]*ds <= dss)
        g1       = 1.0 * len(self.local_info[3]) - sum(np.array(d_one) == 1)
        g2       = np.linalg.norm(x - self.local_info[4][self.local_info[0]]) - self.local_info[1]
        out['G'] = np.column_stack([g1, g2])


class VCBO(AbstractOptimizer):
    # def __init__(self, dims, radius, dim_delta, scale, func, func_name,
    #              var_lb=np.array([-1]*10), var_ub=np.array([1]*10), init_num=50):
    def __init__(self, 
            space       : DesignSpace,
            rand_sample : int = 50, 
            radius      : float = None, 
            scale       : float = None, 
            dim_delta   : float = None, 
            ):
        self.space       = space
        self.rand_sample = rand_sample
        self.dims        = space.num_paras
        self.var_lb      = space.opt_lb.numpy()
        self.var_ub      = space.opt_ub.numpy()
        self.radius      = 0.4 * np.linalg.norm(self.var_ub - self.var_lb) if radius is None else radius
        self.scale       = 0.8 if scale is None else scale
        self.dim_delta   = 0.3 * np.mean(self.var_ub - self.var_lb) if dim_delta is None else dim_delta
        self.X           = np.zeros((0, self.dims))
        self.Y           = []
        self.shrink      = False

        for k, p in self.space.paras.items():
            assert not p.is_discrete_after_transform, f"VCBO currently only accept continuous parameters, invalid parameter {k}"

        self.random_state = np.random.RandomState(42)
        self.mask = ["real" for i in range(0, self.dims)]
        self.sampling = MixedVariableSampling(self.mask,   {"real": get_sampling("real_random"),
                                                            "int": get_sampling("int_random")})
        self.crossover = MixedVariableCrossover(self.mask, {"real": get_crossover("real_sbx", prob=1.0, eta=3.0),
                                                            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)})
        self.mutation = MixedVariableMutation(self.mask,   {"real": get_mutation("real_pm", eta=3.0),
                                                            "int": get_mutation("int_pm", eta=3.0)})


    @property
    def best_x(self) -> pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            best_x = torch.FloatTensor(self.X[[np.argmin(self.Y)]])
            xc = best_x[:, :self.space.num_numerical]
            xe = best_x[:, self.space.num_numerical:]
            return self.space.inverse_transform(xc, xe)

    @property
    def best_y(self) -> float:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return np.array(self.Y).min()

    def suggest(self, n_suggestions : int = 1) -> pd.DataFrame:
        assert n_suggestions == 1
        if self.X is None or self.X.shape[0] < self.rand_sample:
            return self.space.sample(n_suggestions)
        else:
            x_opt    = self.search()
            x_tensor = torch.FloatTensor(x_opt)
            x_para   = self.space.inverse_transform(x_tensor, None) # from tensor to DataFraem
            return x_para

    def observe(self, param : pd.DataFrame, y : np.ndarray):
        Xc, _  = self.space.transform(param)
        self.X = np.vstack([self.X, Xc.numpy()])
        self.Y = self.Y + y.reshape(-1).tolist()

    def random_sample(self, d_ball):
        output_flag = 0
        while output_flag == 0:
            pone = self.random_state.uniform(self.random_bound_lb, self.random_bound_ub)
            ds = np.linalg.norm(pone - self.X[self.local_constraints[0]])
            if ds >= d_ball:
                continue
            d_one = []
            for k in self.indices_nb_sites:
                dss = np.linalg.norm(pone - self.X[k])
                d_one.append(self.local_constraints[2] * ds <= dss)
            if sum(np.array(d_one) == 1) < 1.0 * len(self.indices_nb_sites):
                continue
            output_flag = 1
        return pone

    def check_shrink_state(self):
        if self.shrink:
            self.radius = self.radius * 0.8
            self.dim_delta = self.dim_delta * 0.8
            self.shrink = False

    def construct_voronoi_cell(self, tree, ref_point, print_or_not=False):
        self.local_constraints = [ref_point, self.radius, self.scale]
        # points in radius-ball and bad nb sites
        indices = tree.query_ball_point(self.X[self.local_constraints[0]], self.local_constraints[1])
        dmean_ball = np.mean([np.linalg.norm(self.X[idx] - self.X[self.local_constraints[0]]) for idx in indices])
        self.indices_nb_sites = [index for index in indices if
                                 np.linalg.norm(self.X[index] - self.X[self.local_constraints[0]]) > dmean_ball]

        # local points in constrained area
        self.indices_local_points = []
        for idx in indices:
            if idx in self.indices_nb_sites:
                continue
            d_one = []
            for k in self.indices_nb_sites:
                ds = np.linalg.norm(self.X[idx] - self.X[self.local_constraints[0]])
                dss = np.linalg.norm(self.X[idx] - self.X[k])
                d_one.append(ds <= dss)
            if sum(np.array(d_one) == 1) >= 1.0 * len(self.indices_nb_sites):
                if print_or_not:
                    print(idx, self.Y[idx])
                self.indices_local_points.append(idx)
        if print_or_not:
            print(len(self.indices_local_points))
            print('-----------nb sites-------------')
            print(len(self.indices_nb_sites))
            for k in range(len(self.indices_nb_sites)):
                print(self.indices_nb_sites[k], self.Y[self.indices_nb_sites[k]])

        self.random_bound_lb = np.array([np.max([self.var_lb[idx], self.X[self.local_constraints[0]][idx] - self.dim_delta])
                                    for idx in range(self.dims)])
        self.random_bound_ub = np.array([np.min([self.var_ub[idx], self.X[self.local_constraints[0]][idx] + self.dim_delta])
                                    for idx in range(self.dims)])

    def gpbo_in_vcell(self, gp_model, d_ball, kappa = 1.0, noise_level = 0.1):
        xi = 0.0

        algorithm = get_algorithm('ga', pop_size=50, sampling=self.sampling, crossover=self.crossover, mutation=self.mutation,
                       eliminate_duplicates=True)
        prob = MyProblem_gplocal(gp_model, np.min(self.Ys), self.local_constraints + [self.indices_local_points, self.X],
                                 var_num=self.dims, lb=self.random_bound_lb, ub=self.random_bound_ub, kappa=kappa, xi=xi,
                                 noise_level=noise_level)
        res = minimize(prob, algorithm, ('n_gen', 100), seed=1, pf=None, save_history=False, verbose=False)

        if res.X is None:
            x_opt = self.random_sample(d_ball)
        else:
            check = np.array([int((res.X == self.X[knum]).all()) for knum in range(len(self.X))])
            if sum(check == 1) >= 1:
                x_opt = self.random_sample(d_ball)
            else:           
                x_opt = np.asarray(res.X).astype(float)
        return x_opt

    def search(self):
        # # selected best one and construct voronoi cell
        min_one = np.argmin(self.Y)
        tree = KDTree(self.X)
        self.check_shrink_state()
        self.construct_voronoi_cell(tree, min_one)
        self.Xs = np.array([self.X[idx] for idx in self.indices_local_points + self.indices_nb_sites]).astype('float64')
        self.Ys_real = np.array([self.Y[idx] for idx in self.indices_local_points + self.indices_nb_sites]).astype('float64').reshape(-1, 1)

        # simple y transformation
        self.Ys = self.Ys_real.copy() - np.mean(self.Ys_real)
        Yrange = np.max(self.Ys_real) - np.min(self.Ys_real)
        self.Yfactor = 5 / Yrange if Yrange != 0 else 1
        self.Ys = self.Ys * self.Yfactor

        if len(self.indices_local_points) == 0:
            d_ball = self.radius / 2
        else:
            d_max = np.max([np.linalg.norm(self.X[idx] - self.X[self.local_constraints[0]])
                            for idx in self.indices_local_points if idx not in self.indices_nb_sites])
            d_ball = self.radius / 2 if len(self.indices_local_points) < 10 else d_max

        shrink_threshold = 30

        if len(self.indices_local_points) > shrink_threshold:
            self.shrink = True
            print('-----------will shrink------------r{}_dd{}'.format(self.radius, self.dim_delta))

        kappa_list       = [1.0]
        noise_level_list = [0, 0.2, 0.4]
        combinations     = [[k_one, n_one] for k_one in kappa_list for n_one in noise_level_list]
        algo_idx         = (self.X.shape[0] - self.rand_sample) % 4
        if algo_idx < 3:
            kappa, noise_level = combinations[algo_idx]
            gp_model = self.gp_model_fit()
            if gp_model is not None:
                x_opt = self.gpbo_in_vcell(gp_model, d_ball, kappa, noise_level)
            else:
                x_opt = self.random_sample(d_ball)
        else:
            x_opt = self.random_sample(d_ball)
        return x_opt.reshape(-1, self.dims)

    def gp_model_fit(self):
        gp_model = get_model('gpy', torch.from_numpy(self.Xs).shape[1], 0,
                             torch.from_numpy(self.Ys).reshape(-1, 1).shape[1],
                             verbose=False, num_epochs=100)
        try:
            gp_model.fit(torch.FloatTensor(self.Xs), None, torch.FloatTensor(self.Ys).reshape(-1, 1))
        except gpytorch.utils.errors.NanError:
            gp_model = None
        except RuntimeError:
            gp_model = None

        return gp_model
