# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch

from mcbo.search_space import SearchSpace


class TaskBase(ABC):
    """ Abstract class to define optimization (** MINIMISATION **) tasks """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._n_bb_evals = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            A string correponding to the name of the task
        """
        return 'Task Name'

    @abstractmethod
    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """
        Function to compute the problem specific black-box function.

        Args:
            x: dataframe containing the points at which the black-box should be evaluated.
               Shape: (batch_size, num_dims), where num_dims is the dimensionality of the problem and batch_size is the
               batch

        Returns:
             2D numpy array containing evaluated black-box values at the input x. Shape: (batch_size, 1).
        """
        pass

    @property
    def num_func_evals(self):
        return self._n_bb_evals

    def restart(self):
        self._n_bb_evals = 0

    @property
    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return None

    @abstractmethod
    def get_search_space_params(self) -> List[Dict[str, Any]]:
        pass

    def get_search_space(self, dtype: torch.dtype = torch.float64) -> SearchSpace:
        return SearchSpace(params=self.get_search_space_params(), dtype=dtype)

    def increment_n_evals(self, n: int):
        self._n_bb_evals += n

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        self.increment_n_evals(n=len(x))
        return self.evaluate(x.copy())
