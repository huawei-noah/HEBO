# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Optional, List, Callable, Dict, Any, Union

import numpy as np
import pandas as pd
import torch

from mcbo.optimizers.optimizer_base import OptimizerNotBO
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.distance_metrics import hamming_distance
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class SimulatedAnnealing(OptimizerNotBO):
    color_1: str = get_color(ind=9, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return SimulatedAnnealing.color_1

    @staticmethod
    def get_color() -> str:
        return SimulatedAnnealing.get_color_1()

    @property
    def name(self) -> str:
        if self.fixed_tr_manager is not None:
            name = 'Tr-based Simulated Annealing'
        else:
            name = 'Simulated Annealing'
        return name

    @property
    def tr_name(self) -> str:
        return "no-tr"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 init_temp: float = 100.,
                 tolerance: int = 1000,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 max_n_perturb_num: int = 20,
                 neighbourhood_ball_transformed_radius: int = .1,
                 dtype: torch.dtype = torch.float64,
                 ):
        """
        Args:
            fixed_tr_manager: the SA will evolve within the TR defined by the fixed_tr_manager
            store_observations: whether to store observed points
        """
        assert search_space.num_permutation == 0, \
            'Simulated Annealing is currently not implemented for permutation variables'

        self.fixed_tr_manager = fixed_tr_manager
        super(SimulatedAnnealing, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )

        assert len(self.out_constr_dims) == 0, "Do not support multi-obj / constraints yet"
        assert len(self.obj_dims) == 1, "Do not support multi-obj / constraints yet"

        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.discrete_choices = get_discrete_choices(search_space)
        self.max_n_perturb_num = max_n_perturb_num
        self.neighbourhood_ball_transformed_radius = neighbourhood_ball_transformed_radius

        self.init_temp = init_temp
        self.tolerance = tolerance
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions

        self.temp = self.init_temp
        if self.fixed_tr_manager:
            self.x_init = self.search_space.inverse_transform(self.fixed_tr_manager.center.unsqueeze(0))
        else:
            self.x_init = self.sample_input_valid_points(n_points=1, point_sampler=self.search_space.sample)

        self._current_x = None
        self._current_y = None

        # For stability
        self.MAX_EXPONENT = 0  # 0 As probability can't be larger than 1
        self.MIN_EXPONENT = -12  # exp(-12) ~ 6e-6 < self.MIN_PROB
        self.MAX_PROB = 1.
        self.MIN_PROB = 1e-5

    def set_x_init(self, x: pd.DataFrame):
        self.x_init = x

    def restart(self):
        self._restart()

        self._current_x = None
        self._current_y = None
        self.temp = self.init_temp
        if self.fixed_tr_manager:
            self.x_init = self.search_space.inverse_transform(self.fixed_tr_manager.center)
        else:
            self.x_init = self.sample_input_valid_points(n_points=1, point_sampler=self.search_space.sample)

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims
        assert self.fixed_tr_manager is None, "Cannot initialize if a fixed tr_manager is set"

        x = self.search_space.transform(x)
        self.x_init = self.x_init[len(x):]

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # update best fx
        self.update_best(x_transf=x, y=y)

        best_idx = self.get_best_y_ind(y=y)
        best_y = y[best_idx].clone()

        if self._current_x is None or self._current_y is None:
            self._current_y = best_y
            self._current_x = x[best_idx: best_idx + 1]

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        if n_remaining and len(self.x_init):
            n = min(n_remaining, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[idx: idx + n]
            self.x_init = self.x_init.drop(self.x_init.index[[i for i in range(idx, idx + n)]]).reset_index(drop=True)

            idx += n
            n_remaining -= n

        if n_remaining and self._current_x is None:
            if self.fixed_tr_manager:  # sample within TR
                point_sampler = lambda n_points: self.search_space.inverse_transform(
                    sample_numeric_and_nominal_within_tr(
                        x_centre=self.fixed_tr_manager.center,
                        search_space=self.search_space,
                        tr_manager=self.fixed_tr_manager,
                        n_points=n_points,
                        numeric_dims=self.numeric_dims,
                        discrete_choices=self.discrete_choices,
                        max_n_perturb_num=self.max_n_perturb_num,
                        model=None,
                        return_numeric_bounds=False
                    )
                )
            else:
                point_sampler = self.search_space.sample
            x_next.iloc[idx: idx + n_remaining] = self.sample_input_valid_points(n_points=n_remaining,
                                                                                 point_sampler=point_sampler)
            idx += n_remaining
            n_remaining -= n_remaining

        if n_remaining:
            assert self._current_x is not None
            current_x = self._current_x.clone() * torch.ones((n_remaining, self._current_x.shape[1])).to(
                self._current_x)

            def point_sampler(n_points: int) -> pd.DataFrame:
                # create tensor with the good shape
                neighbors = self.search_space.transform(self.search_space.sample(n_points))

                # sample neighbor for nominal dims
                neighbors[:, self.search_space.nominal_dims] = self.sample_unseen_nominal_neighbour(
                    current_x[:, self.search_space.nominal_dims])

                # TODO: check this  --> do we make sure everything is in [0, 1] in transformed space?
                # sample neighbor for numeric dims
                dim_arrays = [self.search_space.disc_dims, self.search_space.cont_dims]
                for dim_array in dim_arrays:
                    if len(dim_array) > 0:
                        noise = torch.randn((n_points, len(dim_array))).to(
                            self._current_x) * self.neighbourhood_ball_transformed_radius
                        # project back to the state space
                        clip_lb = 0
                        clip_ub = 1
                        if self.fixed_tr_manager:  # make sure neighbor is in TR
                            clip_lb = torch.maximum(torch.zeros(len(dim_array)).to(neighbors),
                                                    self.fixed_tr_manager.center[dim_array] -
                                                    self.fixed_tr_manager.radii[
                                                        'numeric'])
                            clip_ub = torch.minimum(torch.ones(len(dim_array)).to(neighbors),
                                                    self.fixed_tr_manager.center[dim_array] +
                                                    self.fixed_tr_manager.radii[
                                                        'numeric'])
                        neighbors[:, dim_array] = torch.clip(current_x[:, dim_array] + noise, clip_lb, clip_ub)
                return self.search_space.inverse_transform(neighbors)

            x_next.iloc[idx: idx + n_remaining] = self.sample_input_valid_points(
                point_sampler=point_sampler, n_points=n_remaining, allow_repeat=self.allow_repeating_suggestions
            )

        return x_next

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        self.update_best(x_transf=x.clone(), y=y)
        idx = self.get_best_y_ind(y=y)
        candidate_best_y = y[idx].clone()
        # update best fx
        if self._current_y is None:
            self._current_x = x[idx: idx + 1].clone()
            self._current_y = y[idx].clone()
        else:
            self.temp *= 0.8

            if self.is_better_than_current(current_y=self._current_y, new_y=candidate_best_y):
                self._current_x = x[idx: idx + 1].clone()
                self._current_y = candidate_best_y
            else:
                assert self.n_objs == 1 and self.n_constrs == 0, (self.n_objs, self.n_constrs)
                gap = candidate_best_y[0].item() - self._current_y[0].item()
                exponent = np.clip(- gap / self.temp, self.MIN_EXPONENT, self.MAX_EXPONENT)
                p = np.clip(np.exp(exponent), self.MIN_PROB, self.MAX_PROB)
                z = np.random.rand()

                if z < p:
                    self._current_x = x[idx: idx + 1]
                    self._current_y = candidate_best_y

    def sample_unseen_nominal_neighbour(self, x_nominal: torch.Tensor):

        if not self.allow_repeating_suggestions:
            x_observed = self.data_buffer.x[:, self.search_space.nominal_dims]

        single_sample = False

        if x_nominal.ndim == 1:
            x_nominal = x_nominal.view(1, -1)

            single_sample = True

        x_nominal_neighbour = x_nominal.clone()

        for idx in range(len(x_nominal)):
            done = False
            tol = self.tolerance
            while not done:
                x = x_nominal[idx]
                # randomly choose a nominal variable
                if self.fixed_tr_manager and hamming_distance(
                        self.fixed_tr_manager.center[self.search_space.nominal_dims].to(x), x,
                        normalize=False) >= self.fixed_tr_manager.get_nominal_radius():
                    # choose a dim that won't suggest a neighbor out of the TR
                    var_idx = np.random.choice(
                        [i for i, d in enumerate(self.search_space.nominal_dims) if
                         x[i] != self.fixed_tr_manager.center[d]])
                else:
                    var_idx = np.random.randint(low=0, high=self.search_space.num_nominal)
                choices = [j for j in range(int(self.search_space.nominal_lb[var_idx]),
                                            int(self.search_space.nominal_ub[var_idx]) + 1) if
                           j != x_nominal[idx, var_idx]]

                x[var_idx] = np.random.choice(choices)

                tol -= 1

                if self.allow_repeating_suggestions:
                    done = True
                elif not (x.unsqueeze(0) == x_observed).all(1).any():
                    done = True
                elif tol == 0:
                    if self.fixed_tr_manager:
                        x = sample_numeric_and_nominal_within_tr(x_centre=self.fixed_tr_manager.center,
                                                                 search_space=self.search_space,
                                                                 tr_manager=self.fixed_tr_manager,
                                                                 n_points=1,
                                                                 numeric_dims=self.numeric_dims,
                                                                 discrete_choices=self.discrete_choices,
                                                                 max_n_perturb_num=self.max_n_perturb_num,
                                                                 model=None,
                                                                 return_numeric_bounds=False)[0]
                    else:
                        x = self.search_space.transform(self.search_space.sample(1))[0]
                    done = True
                    x = x[self.search_space.nominal_dims]

            x_nominal_neighbour[idx] = x

        if single_sample:
            x_nominal_neighbour = x_nominal_neighbour.view(-1)

        return x_nominal_neighbour

    def fill_field_after_pkl_load(self, search_space: SearchSpace, **kwargs):
        """ As some elements are not pickled, need to reinstantiate them """
        self.search_space = search_space

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        to_remove = ["search_space"]  # fields to remove when pickling this object
        for attr in to_remove:
            del d[attr]
        return d
