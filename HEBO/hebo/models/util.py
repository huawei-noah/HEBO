# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn, FloatTensor, LongTensor

def filter_nan(x : FloatTensor, xe : LongTensor, y : FloatTensor, keep_rule = 'any') -> (FloatTensor, LongTensor, FloatTensor):
    assert x  is None or torch.isfinite(x).all()
    assert xe is None or torch.isfinite(xe).all()
    assert torch.isfinite(y).any(), "No valid data in the dataset"

    if keep_rule == 'any':
        valid_id = torch.isfinite(y).any(dim = 1)
    else:
        valid_id = torch.isfinite(y).all(dim = 1)
    x_filtered  = x[valid_id]  if x  is not None else None
    xe_filtered = xe[valid_id] if xe is not None else None
    y_filtered  = y[valid_id]
    return x_filtered, xe_filtered, y_filtered

def construct_hidden(dim, num_layers, num_hiddens, act = nn.ReLU()) -> nn.Module:
    layers = [nn.Linear(dim, num_hiddens), act]
    for i in range(num_layers - 1):
        layers.append(nn.Linear(num_hiddens, num_hiddens))
        layers.append(act)
    return nn.Sequential(*layers)
