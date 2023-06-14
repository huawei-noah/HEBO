# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space import SearchSpace
from typing import Optional, Union, List

from mcbo.utils.general_utils import copy_tensor


def get_discrete_choices(search_space: SearchSpace,
                         selected_dims: Optional[Union[List[int], np.ndarray]] = None) -> Union[
    torch.Tensor, List[torch.Tensor]]:
    if selected_dims is None:
        selected_dims = search_space.disc_dims
    n_choices_per_dim = [int(ub - lb + 1) for ub, lb in
                         zip(search_space.opt_ub[selected_dims], search_space.opt_lb[selected_dims])]

    if len(n_choices_per_dim) == 0 or np.all(n_choices_per_dim == n_choices_per_dim[0]):
        return torch.tensor(np.array([np.linspace(0, 1, n_choices) for n_choices in n_choices_per_dim]),
                            dtype=search_space.dtype)
    else:
        return [torch.tensor(np.linspace(0, 1, n_choices), dtype=search_space.dtype) for n_choices in
                n_choices_per_dim]


def round_discrete_vars(x: torch.FloatTensor, discrete_dims: list,
                   choices: Union[torch.FloatTensor, List[torch.Tensor]]) -> torch.FloatTensor:
    if len(discrete_dims):
        x_ = copy_tensor(x)
        ndim = x_.ndim
        if ndim == 1:
            x_ = x_.view(1, -1)
        if type(choices) == torch.Tensor:
            delta_x = torch.abs(x_[:, discrete_dims, None] - choices[None, ...])
            indices = torch.argmin(delta_x, axis=2)
            x_discrete = torch.gather(torch.repeat_interleave(choices[None, ...], len(x), dim=0), dim=2,
                                      index=indices[..., None]).squeeze(dim=-1)
        elif type(choices) == list and type(choices[0]) == torch.Tensor:
            delta_x = [torch.abs(x_[:, discrete_dims[i], None] - choices[i][None, ...].to(device=x_.device)) for i in range(len(discrete_dims))]
            indices = [torch.argmin(delta_x_i, axis=1) for delta_x_i in delta_x]
            x_discrete = torch.vstack([choices[i][indices[i]] for i in range(len(choices))]).T.to(x_)
        else:
            raise ValueError((choices, type(choices)))
        x_[:, discrete_dims] = x_discrete

        if ndim == 1:
            x_ = x_.squeeze(dim=0)

        return x_
    else:
        return x
