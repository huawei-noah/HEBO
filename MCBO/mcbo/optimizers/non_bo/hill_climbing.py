# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional, List, Callable, Dict, Union

import numpy as np
import pandas as pd
import torch

from mcbo.optimizers.optimizer_base import OptimizerNotBO
from mcbo.search_space import SearchSpace
from mcbo.utils.plot_resource_utils import get_color, COLORS_SNS_10


class HillClimbing(OptimizerNotBO):
    color_1: str = get_color(ind=7, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return HillClimbing.color_1

    @staticmethod
    def get_color() -> str:
        return HillClimbing.get_color_1()

    @property
    def name(self) -> str:
        return 'Hill Climbing'

    @property
    def tr_name(self) -> str:
        return "no-tr"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 tolerance: int = 1000,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 neighbourhood_ball_transformed_radius: int = .1,
                 dtype: torch.dtype = torch.float64,
                 ):
        """
        Args:
             neighbourhood_ball_normalised_radius: in the transformed space, numerical dims are mutated by sampling
                                                      a Gaussian perturbation with std this value
        """
        super(HillClimbing, self).__init__(
            search_space=search_space,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            dtype=dtype
        )

        self.neighbourhood_ball_transformed_radius = neighbourhood_ball_transformed_radius
        self.tolerance = tolerance
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions

        self.x_init = self.sample_input_valid_points(n_points=1, point_sampler=self.search_space.sample)

        self._current_x = None
        self._current_y = None

    def set_x_init(self, x: pd.DataFrame):
        self.x_init = x

    def restart(self):
        self._restart()
        self._current_x = None
        self._current_y = None
        self.x_init = self.sample_input_valid_points(n_points=1, point_sampler=self.search_space.sample)

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x_transf = self.search_space.transform(x)
        self.x_init = self.x_init[len(x_transf):]

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x_transf.clone(), y.clone())

        # update best fx
        self.update_best(x_transf=x_transf, y=y)

        best_idx = self.get_best_y_ind(y=y)
        best_y = y[best_idx]

        if self._current_x is None or self._current_y is None:
            self._current_y = best_y
            self._current_x = x_transf[best_idx: best_idx + 1]

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
            x_next.iloc[idx: idx + n_remaining] = self.sample_input_valid_points(
                n_points=n_remaining, point_sampler=self.search_space.sample
            )
            idx += n_remaining
            n_remaining -= n_remaining

        if n_remaining:
            assert self._current_x is not None

            def point_sampler(n_points: int):
                current_x = self._current_x.clone() * torch.ones((n_remaining, self._current_x.shape[1])).to(
                    self._current_x)

                # create tensor with the good shape
                neighbors = self.search_space.transform(self.search_space.sample(n_remaining))

                # sample neighbor for nominal dims
                neighbors[:, self.search_space.nominal_dims] = self.sample_unseen_nominal_neighbour(
                    current_x[:, self.search_space.nominal_dims])

                # TODO: check this  --> do we make sure everything is in [0, 1] in transformed space?
                # sample neighbor for numeric dims
                dim_arrays = [self.search_space.disc_dims, self.search_space.cont_dims]
                for dim_array in dim_arrays:
                    if len(dim_array) > 0:
                        noise = torch.randn((n_remaining, len(dim_array))).to(
                            self._current_x) * self.neighbourhood_ball_transformed_radius
                        # project back to the state space
                        clip_lb = 0
                        clip_ub = 1
                        neighbors[:, dim_array] = torch.clip(current_x[:, dim_array] + noise, clip_lb, clip_ub)
                return self.search_space.inverse_transform(neighbors)

            x_next.iloc[idx: idx + n_remaining] = self.sample_input_valid_points(
                n_points=n_remaining,
                point_sampler=point_sampler
            )

        return x_next

    def method_observe(self, x, y):

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # update best fx
        self.update_best(x_transf=x, y=y)

        idx = self.get_best_y_ind(y=y)
        y_ = y[idx].clone()
        if self._current_y is None or self.is_better_than_current(current_y=self._current_y, new_y=y_):
            self._current_x = x[idx: idx + 1].clone()
            self._current_y = y_


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
                    x = self.search_space.transform(self.search_space.sample(1))[0]
                    done = True
                    x = x[self.search_space.nominal_dims]

            x_nominal_neighbour[idx] = x

        if single_sample:
            x_nominal_neighbour = x_nominal_neighbour.view(-1)

        return x_nominal_neighbour
