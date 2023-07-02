# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from abc import abstractmethod
from typing import Optional, Callable, List, Dict, Union

import numpy as np
import pandas as pd
import torch

from mcbo.search_space import SearchSpace
from mcbo.utils.constraints_utils import input_eval_from_origx, input_eval_from_transfx, \
    sample_input_valid_points
from mcbo.utils.data_buffer import DataBuffer
from mcbo.utils.general_utils import filter_nans


class OptimizerBase(ABC):

    @staticmethod
    def get_linestyle() -> str:
        return "-"

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def get_name(self, no_alias: bool = False) -> str:
        return self.name

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def acq_opt_name(self) -> str:
        pass

    @property
    @abstractmethod
    def acq_func_name(self) -> str:
        pass

    @property
    @abstractmethod
    def tr_name(self) -> str:
        pass

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 dtype: torch.dtype = torch.float64,
                 ):
        """
        Args:
            search_space: optimization search space
            dtype: tensor type
            input_constraints: list of funcs taking a point as input and outputting whether the point
                                       is valid or not
        """
        assert dtype in [torch.float32, torch.float64]

        self.dtype = dtype
        self.search_space = search_space
        self.input_constraints = input_constraints

        self._best_x = None
        self.best_y = None

        self.data_buffer = DataBuffer(num_dims=self.search_space.num_dims, num_out=1, dtype=self.dtype)

    @abstractmethod
    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        """
        Function used to suggest next query points.
        Should return a pandas dataframe with shape (n_suggestions, D) where
        D is the dimensionality of the problem. The column dtype may mismatch expected dtype (e.g. float for int)

        Args:
            n_suggestions: number of suggestions

        Returns:
             a DataFrame of suggestions
        """
    pass

    def suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        """
        **DO NOT OVERRIDE THIS, OVERRIDE `method_suggest`**

        Function used to suggest next query points. Should return a pandas dataframe with shape (n_suggestions, D) where
        D is the dimensionality of the problem.

        Args:
            n_suggestions: number of points to suggest
            
        Returns:
            dataframe of suggested points.
        """
        suggestions = self.method_suggest(n_suggestions)
        # Convert the dtype of each column to proper dtype
        sample = self.search_space.sample(1)
        return suggestions.astype({column_name: sample.dtypes[column_name] for column_name in sample})

    @abstractmethod
    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """
        Function used to store observations and to conduct any algorithm-specific computation.

        Args:
            x: points in search space
            y: 2-d array of black-box values
        """
        pass

    def observe(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """
        ** DO NOT OVERRIDE THIS, OVERRIDE `method_observe` **

        Function used to store observations and to conduct any algorithm-specific computation.

        Args:
            x: points in search space
            y: 2-d array of black-box values
        """
        assert len(x) == len(y)
        x, y = filter_nans(x=x, y=y)

        if len(x) == 0:
            return
        return self.method_observe(x=x, y=y)

    @abstractmethod
    def restart(self):
        """
        Function used to restart the internal state of the optimizer between different runs on the same task.
        Returns:
        """
        pass

    @abstractmethod
    def set_x_init(self, x: pd.DataFrame) -> None:
        """
        Function to set query points that should be suggested during random exploration

        Args:
            x: points to set as initial points
        """
        pass

    @abstractmethod
    def initialize(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """
        Function used to initialise an optimizer with a dataset of observations

        Args:
            x: points in search space given as a DataFrame
            y: (2D-array) values
        Returns:
        """
        pass

    @property
    def best_x(self):
        if self.best_y is not None:
            return self.search_space.inverse_transform(self._best_x)

    def _restart(self):
        self._best_x = None
        self.best_y = None

        self.data_buffer.restart()

    def input_eval_from_transfx(self, transf_x: torch.Tensor) -> np.ndarray:
        """
        Evaluate the boolean constraint function on a set of transformed inputs

        Returns:
            Array of `number of input points \times number of input constraint` booleans
                specifying at index `(i, j)` if input point `i` is valid regarding constraint function `j`        """
        return input_eval_from_transfx(transf_x=transf_x, search_space=self.search_space,
                                               input_constraints=self.input_constraints)

    def input_eval_from_origx(self, x: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Evaluate the boolean constraint function on a set of non-transformed inputs

        Args:
            x: can contain several input points as a Dataframe, can also be given as a single Dict {var_name: var_value}

        Returns:
            Array of `number of input points \times number of input constraint` booleans
                specifying at index `(i, j)` if input point `i` is valid regarding constraint function `j`
        """
        return input_eval_from_origx(x=x, input_constraints=self.input_constraints)

    def sample_input_valid_points(self, n_points: int, point_sampler: Callable[[int], pd.DataFrame],
                                          max_trials: int = 100, allow_repeat: bool = True) -> pd.DataFrame:
        """
        Sample valid points from the original space that fulfill the input constraints

        Args:
            n_points: number of points desired
            point_sampler: function that can be used to sample points
            max_trials: max number of trials
            allow_repeat: whether the same point can be suggested several time

        Returns:
            samples: `n_points` valid samples
        """
        return sample_input_valid_points(
            n_points=n_points,
            point_sampler=point_sampler,
            input_constraints=self.input_constraints,
            allow_repeat=allow_repeat,
            max_trials=max_trials
        )


class OptimizerNotBO(OptimizerBase):

    @property
    def model_name(self) -> str:
        return "no-model"

    @property
    def acq_opt_name(self) -> str:
        return "no-acq-opt"

    @property
    def acq_func_name(self) -> str:
        return "no-acq-func"

