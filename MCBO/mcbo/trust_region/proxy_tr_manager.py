# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union, Optional, List, Callable, Dict

import numpy as np
import pandas as pd
import torch

from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.utils.data_buffer import DataBuffer


class ProxyTrManager(TrManagerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 obj_dims: Union[List[int], np.ndarray],
                 out_constr_dims: Union[List[int], np.ndarray],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 dtype: torch.dtype = torch.float64,
                 ):
        super(ProxyTrManager, self).__init__(
            search_space=search_space,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            dtype=dtype
        )

    def restart(self) -> None:
        pass

    def adjust_tr_radii(self, y: torch.Tensor, **kwargs):
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
            best_y: Used for evaluating some acquisition functions such as the Expected Improvement acquisition
            kwargs:
        """

        pass
