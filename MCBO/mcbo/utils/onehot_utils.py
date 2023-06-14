# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch


def onehot_encode(x: torch.Tensor, ub: torch.Tensor, num_onehot_vars: int) -> torch.Tensor:
    assert x.ndim == 2
    assert ub.ndim == 1
    assert x.shape[1] == ub.shape[0]

    n_points = x.shape[0]
    n_vars = x.shape[1]

    indices = x.clone()
    indices[:, 1:] = indices[:, 1:] + torch.cumsum(ub + 1, dim=0)[:-1].to(x)
    indices = indices.to(dtype=torch.long)

    x_ = torch.zeros((n_points, num_onehot_vars)).to(x)
    for i in range(n_vars):
        x_.scatter_(index=indices[:, i:i + 1], dim=1, value=1)

    return x_


def onehot_decode(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    assert ub.ndim == 1
    assert x.shape[1] == ub.shape[0]

    indices = torch.cat([torch.arange(ub[i] + 1) for i in range(len(ub))]).view(1, -1).to(x)
    indices = indices * torch.ones_like(x)

    return indices[x.to(dtype=torch.bool)].view(x.shape[0], -1)
