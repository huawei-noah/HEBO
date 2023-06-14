# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from mcbo.global_settings import MIN_LEN_CUDA
from mcbo.utils.data_buffer import DataBuffer
from mcbo.models import ModelBase


def move_model_to_device(model: ModelBase, data_buffer: DataBuffer, target_device: torch.device):
    if model.supports_cuda and torch.cuda.is_available():

        if len(data_buffer) < MIN_LEN_CUDA:
            device = torch.device('cpu')
        else:
            device = target_device

        model.to(device=device, dtype=model.dtype)


def add_hallucinations_and_retrain_model(model: ModelBase, x_next: torch.Tensor):
    # Grab model's internal dataset
    x, y = model.x, model._y

    # Append x and mean prediction for y to dataset
    if x is None:
        x = x_next.unsqueeze(0)
    else:
        x = torch.cat((x, x_next.unsqueeze(0).to(x)))  # append x to model's dataset
    with torch.no_grad():
        y_next_mean, _ = model.predict(x_next.unsqueeze(0))
    if y is None:
        y = y_next_mean.to(y)
    else:
        y = torch.cat((model.fit_y_to_y(fit_y=y), y_next_mean.to(y)))

    # Call the pre fit method
    model.pre_fit_method(x, y)

    # Retrain model
    model.fit(x, y)
