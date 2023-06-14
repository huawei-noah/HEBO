# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional, List, Callable, Dict

import numpy as np
import pandas as pd
import torch

from mcbo.optimizers.optimizer_base import OptimizerNotBO
from mcbo.search_space import SearchSpace
from mcbo.utils.plot_resource_utils import COLORS_SNS_10


# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


class RandomSearch(OptimizerNotBO):
    color_1: str = COLORS_SNS_10[5]

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
                 store_observations: bool = False,
                 dtype: torch.dtype = torch.float32
                 ):
        self.store_observations = store_observations
        self.x_init = search_space.sample(0)
        super(RandomSearch, self).__init__(
            search_space=search_space, dtype=dtype, input_constraints=input_constraints
        )

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        # Create a Dataframe that will store the candidates
        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # Return as many points from initialisation as possible
        if len(self.x_init) and n_remaining:
            n = min(n_suggestions, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[[i for i in range(0, n)]]
            self.x_init = self.x_init.drop([i for i in range(0, n)], inplace=False).reset_index(drop=True)

            idx += n
            n_remaining -= n

        if n_remaining:
            x_next[idx: idx + n_remaining] = self.sample_input_valid_points(
                n_points=n_remaining, point_sampler=self.search_space.sample
            )
        return x_next

    def observe(self, x: pd.DataFrame, y: np.ndarray):

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data
        if self.store_observations:
            self.data_buffer.append(x, y)

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations:
            self.data_buffer.append(x.clone(), y.clone())

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

    def set_x_init(self, x: pd.DataFrame):
        self.x_init = x

    def restart(self):
        self._restart()
        self.x_init = self.search_space.sample(0)
