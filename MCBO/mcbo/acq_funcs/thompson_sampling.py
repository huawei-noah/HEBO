# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch

from mcbo.acq_funcs import SingleObjAcqBase
from mcbo.models import ModelBase


class ThompsonSampling(SingleObjAcqBase):

    @property
    def name(self) -> str:
        return "TS"

    def __init__(self):
        super(ThompsonSampling, self).__init__()

    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 **kwargs
                 ) -> torch.Tensor:
        return model.sample_y(x, n_samples=1)[0]
