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
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class RandomSearch(OptimizerNotBO):
    color_1: str = get_color(ind=5, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return RandomSearch.color_1

    @staticmethod
    def get_color() -> str:
        return RandomSearch.get_color_1()

    @property
    def name(self) -> str:
        return 'Random Search'

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
                 store_observations: bool = True,
                 dtype: torch.dtype = torch.float64
                 ):
        """
        Args:
            fixed_tr_manager: the SA will evolve within the TR defined by the fixed_tr_manager
            store_observations: whether to store observed points
        """
        super(RandomSearch, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
        )

        self.store_observations = store_observations
        self.tr_manager = fixed_tr_manager

        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.discrete_choices = get_discrete_choices(search_space)

        self.x_init = self.sample_input_valid_points(
            n_points=1, point_sampler=self.get_tr_point_sampler()
        )

    def get_tr_point_sampler(self) -> Callable[[int], pd.DataFrame]:
        """
        Returns a function taking a number n_points as input and that returns a dataframe containing n_points sampled
        in the search space (within the trust region if a trust region is associated to self)
        """
        if self.tr_manager is not None:
            def point_sampler(n_points: int):
                # Sample points in the trust region of the new centre
                return self.search_space.inverse_transform(
                    sample_numeric_and_nominal_within_tr(
                        x_centre=self.tr_manager.center,
                        search_space=self.search_space,
                        tr_manager=self.tr_manager,
                        n_points=n_points,
                        numeric_dims=self.numeric_dims,
                        discrete_choices=self.discrete_choices,
                        max_n_perturb_num=self.search_space.num_numeric,
                        model=None,
                        return_numeric_bounds=False
                    )
                )
        else:
            # sample a random population
            point_sampler = self.search_space.sample

        return point_sampler

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        # Create a Dataframe that will store the candidates
        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # Return as many points from initialization as possible
        if len(self.x_init) and n_remaining:
            n = min(n_suggestions, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[np.arange(n)]
            self.x_init = self.x_init.drop(
                self.x_init.index[[i for i in range(0, n)]], inplace=False
            ).reset_index(drop=True)

            idx += n
            n_remaining -= n

        if n_remaining:
            x_next[idx: idx + n_remaining] = self.sample_input_valid_points(
                n_points=n_remaining, point_sampler=self.get_tr_point_sampler()
            )
        return x_next

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:

        x_transf = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x_transf) == len(y)

        # Add data to all previously observed data
        if self.store_observations:
            self.data_buffer.append(x_transf, y)

        # update best fx
        self.update_best(x_transf=x_transf, y=y)

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
        if self.store_observations:
            self.data_buffer.append(x_transf.clone(), y.clone())

        # update best fx
        self.update_best(x_transf=x_transf, y=y)

    def set_x_init(self, x: pd.DataFrame):
        self.x_init = x

    def restart(self):
        self._restart()
        self.x_init = self.search_space.sample(0)

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        to_remove = ["search_space"]  # fields to remove when pickling this object
        for attr in to_remove:
            del d[attr]
        return d
