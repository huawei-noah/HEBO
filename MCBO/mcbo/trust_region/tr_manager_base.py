# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Callable, Dict

import numpy as np
import pandas as pd
import torch

from mcbo.search_space import SearchSpace
from mcbo.utils.data_buffer import DataBuffer
from mcbo.utils.multi_obj_constr_utils import get_best_y_ind, is_better_than_current


class TrManagerBase(ABC):

    def __init__(self,
                 search_space: SearchSpace,
                 obj_dims: Union[List[int], np.ndarray],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 dtype: torch.dtype = torch.float64,
                 **kwargs
                 ):
        """
        Args:
            search_space: search space
            obj_dims: dimensions in ys corresponding to objective values to minimize
            out_constr_dims: dimensions in ys corresponding to inequality constraints
            out_upper_constr_vals: values of upper bounds for inequality constraints
            dtype: type of tensor used
        """
        self.radii = {}
        self.min_radii = {}
        self.max_radii = {}
        self.init_radii = {}
        self.variable_types = []
        self._center = None

        if out_constr_dims is None:
            out_constr_dims = []

        self.obj_dims = obj_dims
        self.out_constr_dims = out_constr_dims
        self.out_upper_constr_vals = out_upper_constr_vals

        self.search_space = search_space
        self.data_buffer: DataBuffer = DataBuffer(
            num_dims=search_space.num_dims, dtype=dtype,
            obj_dims=self.obj_dims, out_constr_dims=self.out_constr_dims,
            out_upper_constr_vals=self.out_upper_constr_vals
        )

    def set_center(self, center: Optional[torch.Tensor]):
        if center is None:
            self._center = None
        else:
            assert center.shape[-1] == self.search_space.num_dims, (center.shape[-1], self.search_space.num_dims)
            self._center = center.to(self.search_space.dtype)

    @property
    def center(self) -> Optional[torch.Tensor]:
        if self._center is None:
            return None
        else:
            return self._center.clone()

    def register_radius(self,
                        variable_type: str,
                        min_radius: Union[int, float],
                        max_radius: Union[int, float],
                        init_radius: Union[int, float]
                        ):
        assert min_radius < init_radius <= max_radius

        self.variable_types.append(variable_type)

        self.radii[variable_type] = init_radius
        self.init_radii[variable_type] = init_radius
        self.min_radii[variable_type] = min_radius
        self.max_radii[variable_type] = max_radius

    def append(self, x: torch.Tensor, y: torch.Tensor):
        self.data_buffer.append(x, y)

    def restart_tr(self):
        self.data_buffer.restart()

        for var_type in self.variable_types:
            self.radii[var_type] = self.init_radii[var_type]

        self.set_center(None)

    @abstractmethod
    def restart(self) -> None:
        pass

    @abstractmethod
    def adjust_tr_radii(self, y: torch.Tensor, **kwargs) -> None:
        """
        Function used to update each radius stored in self.radii
        :return:
        """
        pass

    def adjust_tr_center(self, **kwargs):
        """
        Function used to update the TR center
        :return:
        """
        self.set_center(self.data_buffer.best_x)

    @abstractmethod
    def suggest_new_tr(self, n_init: int, observed_data_buffer: DataBuffer,
                       input_constraints: Optional[List[Callable[[Dict], bool]]],
                       **kwargs) -> pd.DataFrame:
        """
        Function used to suggest a new trust region centre and neighbouring points

        Args:
            n_init:
            input_constraints: list of funcs taking a point as input and outputting whether the point
                                       is valid or not
            observed_data_buffer: Data buffer containing all previously observed points
            kwargs:
        """

        pass

    def get_nominal_radius(self) -> float:
        """
        Return the radius associated to nominal variables (note that if there is only one nominal dimension in
        the search space, the radius is always 1)
        """
        if self.search_space.num_nominal == 1:
            return 1
        else:
            assert "nominal" in self.radii
            return self.radii["nominal"]

    def get_best_y_ind(self, y: torch.Tensor) -> int:
        """ Get index of best entry in y, taking into account objective and constraints
            If some entries fulfill the constraints, return the best among them
            Otherwise, return the index of the entry that is the closest to fulfillment
        """
        best_ind = get_best_y_ind(y=y, obj_dims=self.obj_dims, out_constr_dims=self.out_constr_dims,
                                  out_upper_constr_vals=self.out_upper_constr_vals)
        return best_ind

    def is_better_than_current(self, current_y: torch.Tensor, new_y: torch.Tensor) -> bool:
        if torch.any(torch.isnan(new_y)):  # new_y contains NaN so it is not better
            return False
        return is_better_than_current(
            current_y=current_y, new_y=new_y, obj_dims=self.obj_dims,
            out_constr_dims=self.out_constr_dims,
            out_upper_constr_vals=self.out_upper_constr_vals
        )
