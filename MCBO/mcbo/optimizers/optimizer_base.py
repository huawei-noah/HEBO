# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import time
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
from mcbo.utils.multi_obj_constr_utils import get_best_y_ind, get_valid_filter


class OptimizerBase(ABC):
    """ Base class for multi-objective and multi-constrained optimizers
            min(f_1(x), ..., f_n(x))
            s.t. c_1(x) <= lambda_1, ... c_m(x) <= lambda_m
    """

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
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 dtype: torch.dtype = torch.float64,
                 ):
        """
        Args:
            search_space: optimization search space
            dtype: tensor type
            input_constraints: list of funcs taking a point as input and outputting whether the point
                                       is valid or not
            obj_dims: dimensions in ys corresponding to objective values to minimize
            out_constr_dims: dimensions in ys corresponding to inequality constraints
            out_upper_constr_vals: values of upper bounds for inequality constraints
        """
        assert dtype in [torch.float32, torch.float64]

        self.dtype = dtype
        self.search_space = search_space
        self.input_constraints = input_constraints

        # Constraints-related management
        if obj_dims is None:
            obj_dims = np.array([0])
        elif isinstance(obj_dims, list):
            obj_dims = np.array(obj_dims)
        self.obj_dims = obj_dims
        assert len(self.obj_dims) == 1, "Cannot support multi-objective for now"

        if out_constr_dims is not None:
            if isinstance(out_constr_dims, list):
                out_constr_dims = np.array(out_constr_dims)
            assert out_upper_constr_vals is not None
        else:
            out_constr_dims = np.zeros(0, dtype=int)
        self.out_constr_dims = out_constr_dims
        if self.n_constrs == 0:
            out_upper_constr_vals = torch.zeros(0)
        elif out_upper_constr_vals is not None and not isinstance(out_upper_constr_vals, torch.Tensor):
            out_upper_constr_vals = torch.tensor(out_upper_constr_vals).to(dtype=dtype)
        self.out_upper_constr_vals = out_upper_constr_vals

        self.data_buffer = DataBuffer(
            num_dims=self.search_space.num_dims,
            dtype=self.dtype,
            obj_dims=self.obj_dims,
            out_constr_dims=self.out_constr_dims,
            out_upper_constr_vals=self.out_upper_constr_vals
        )

        self._best_x = None
        self.best_y = None

        self.last_suggest_time = None
        self.last_observe_time = None

    @property
    def n_objs(self) -> int:
        return len(self.obj_dims)

    @property
    def n_constrs(self) -> int:
        return len(self.out_constr_dims)

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
        time_ref = time.time()

        suggestions = self.method_suggest(n_suggestions)
        # Convert the dtype of each column to proper dtype
        sample = self.search_space.sample(1)

        self.last_suggest_time = time.time() - time_ref

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
        time_ref = time.time()

        assert len(x) == len(y)
        x, y = filter_nans(x=x, y=y)

        if len(x) > 0:
            self.method_observe(x=x, y=y)

        self.last_observe_time = time.time() - time_ref

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
    def best_x(self) -> Optional[pd.DataFrame]:
        if self.best_y is not None:
            return self.search_space.inverse_transform(self._best_x)
        return None

    def update_best(self, x_transf: torch.Tensor, y: torch.Tensor) -> None:
        """ Update `best_y` and `_best_x` given new suggestions

        Args:
            x_transf: tensor of inputs (in transformed space)
            y: tensor of black-box values
        """
        best_idx = self.get_best_y_ind(y=y)
        best_y = y[best_idx]

        if self.best_y is None or self.is_better_than_current(current_y=self.best_y, new_y=best_y):
            self.best_y = best_y
            self._best_x = x_transf[best_idx: best_idx + 1]

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

    def get_valid_filter(self, y: torch.Tensor) -> torch.Tensor:
        """ Get boolean tensor specifying whether each entry of `y` fulfill the constraints and is not NaN"""
        assert y.ndim == 2, y.shape
        constr_valid_filter = get_valid_filter(
            y=y, out_constr_dims=self.out_constr_dims, out_upper_constr_vals=self.out_upper_constr_vals
        )
        filtr_nan = torch.isnan(y).sum(-1) == 0
        return filtr_nan * constr_valid_filter

    def get_best_y_ind(self, y: torch.Tensor) -> int:
        """ Get index of best entry in y, taking into account objective and constraints
            If some entries fulfill the constraints, return the best among them
            Otherwise, return the index of the entry that is the closest to fulfillment
        """
        assert y.ndim == 2, y.shape
        best_ind = get_best_y_ind(y=y, obj_dims=self.obj_dims, out_constr_dims=self.out_constr_dims,
                                  out_upper_constr_vals=self.out_upper_constr_vals)
        return best_ind

    def get_point_penalty(self, y: torch.Tensor) -> float:
        assert y.ndim == 1, y.shape
        penalties = y[self.out_constr_dims] - self.out_upper_constr_vals.to(y)
        penalty = penalties.max().item()
        return penalty

    def point_is_valid(self, y: torch.Tensor) -> bool:
        assert y.ndim == 1, y.shape
        return self.get_point_penalty(y) <= 0

    @property
    def num_outputs(self) -> int:
        return self.n_constrs + self.n_objs

    def is_better_than_current(self, current_y: torch.Tensor, new_y: torch.Tensor) -> bool:
        """ Check whether new_y is better than current_y  """
        assert len(self.obj_dims) == 1
        assert new_y.shape == current_y.shape == torch.Size([self.n_objs + self.n_constrs]), (
            new_y.shape, current_y.shape, self.n_objs, self.n_constrs)
        if torch.any(torch.isnan(new_y)):
            return False
        if torch.any(torch.isnan(current_y)):
            return True
        if current_y is None:
            return True
        if len(self.out_constr_dims) == 0:
            return new_y[self.obj_dims] < current_y[self.obj_dims]

        # Get penalties of current best and new point
        current_penalty = torch.max(
            current_y[self.out_constr_dims] - self.out_upper_constr_vals.to(current_y)).item()
        new_penalty = torch.max(
            new_y[self.out_constr_dims] - self.out_upper_constr_vals.to(new_y)).item()
        if current_penalty <= 0:  # current is valid: need the new to be valid and better than current best
            return new_penalty <= 0 and new_y[self.obj_dims] < current_y[self.obj_dims]

        # Current best is not valid: need the new to be more valid or equally valid and better than current best
        if new_penalty < current_penalty:
            return True
        return new_penalty == current_penalty and new_y[self.obj_dims] < current_y[self.obj_dims]

    def fill_field_after_pkl_load(self, search_space: SearchSpace, **kwargs):
        """ As some elements are not pickled, need to reinstantiate them """
        self.search_space = search_space

    def __getstate__(self):
        d = dict(self.__dict__)
        to_remove = []  # fields to remove when pickling this object
        for attr in to_remove:
            if attr in d:
                del d[attr]
        return d


class OptimizerNotBO(OptimizerBase, ABC):

    @property
    def model_name(self) -> str:
        return "no-model"

    @property
    def acq_opt_name(self) -> str:
        return "no-acq-opt"

    @property
    def acq_func_name(self) -> str:
        return "no-acq-func"
