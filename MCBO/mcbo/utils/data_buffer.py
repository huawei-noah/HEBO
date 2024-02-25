# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from typing import Optional, Union, List

import numpy as np
import torch

from mcbo.utils.multi_obj_constr_utils import get_best_y_ind


class DataBuffer(ABC):

    def __init__(self, num_dims: int, dtype: torch.dtype,
                 obj_dims: Union[List[int], np.ndarray],
                 out_constr_dims: Union[List[int], np.ndarray, None] = None,
                 out_upper_constr_vals: Optional[torch.Tensor] = None,
                 ):
        """
        Args:
            num_dims: number of dimensions of the inputs stored in the data buffer
            obj_dims: dimensions in ys corresponding to objective values to minimize
            out_constr_dims: dimensions in ys corresponding to inequality constraints
            out_upper_constr_vals: values of upper bounds for inequality constraints
            dtype: type of the data
        """
        super(DataBuffer, self).__init__()
        self.num_dims = num_dims

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

        self.dtype = dtype

        self._x = torch.zeros((0, self.num_dims), dtype=self.dtype)
        self._y = torch.zeros((0, self.num_outputs), dtype=self.dtype)

    @property
    def n_objs(self) -> int:
        return len(self.obj_dims)

    @property
    def n_constrs(self) -> int:
        return len(self.out_constr_dims)

    @property
    def num_outputs(self) -> int:
        return self.n_constrs + self.n_objs

    def append(self, x: torch.Tensor, y: torch.Tensor):
        assert x.ndim == 2
        assert y.ndim == 2, y
        assert len(x) == len(y), (x.shape, y.shape)
        assert y.shape[1] == self.num_outputs

        self._x = torch.cat((self._x, x.clone()), axis=0)
        self._y = torch.cat((self._y, y.clone()), axis=0)

    @property
    def x(self) -> torch.Tensor:
        return self._x.clone()

    @property
    def y(self) -> torch.Tensor:
        return self._y.clone()

    @property
    def best_y(self) -> torch.Tensor:
        """
        Get best y

        Return:
            best_y: tensor of shape (num_outputs,)
        """
        if len(self._y) > 0:
            return self._y[self.best_index].clone()
        else:
            return torch.zeros(self.num_outputs, dtype=self.dtype)

    @property
    def best_index(self) -> int:
        """
        Get best observation index

        Return:
            best_ind: int
        """
        if len(self._y) == 0:
            return 0

        filtr = torch.isnan(self._y).sum(-1) == 0
        ind_filtr = torch.arange(len(filtr), device=filtr.device)[filtr]
        y = self._y[filtr]
        if len(y) > 0:
            return ind_filtr[get_best_y_ind(y=y, obj_dims=self.obj_dims, out_constr_dims=self.out_constr_dims,
                                            out_upper_constr_vals=self.out_upper_constr_vals)]
        return 0

    @property
    def best_x(self) -> Optional[torch.Tensor]:
        filtr = torch.isnan(self._y).sum(-1) == 0
        if not torch.any(filtr):
            return None

        if len(self._y) > 0:
            return self._x[self.best_index]
        else:
            return None

    def __len__(self) -> int:
        return len(self._y)

    def len_without_nan(self) -> int:
        return (torch.isnan(self._y).sum(-1) == 0).sum().item()

    def restart(self):
        self._x = torch.zeros((0, self.num_dims), dtype=self.dtype)
        self._y = torch.zeros((0, self.num_outputs), dtype=self.dtype)
