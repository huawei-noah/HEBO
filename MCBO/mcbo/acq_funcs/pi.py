# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union

import torch
from torch.distributions import Normal

from mcbo.acq_funcs.acq_base import SingleObjAcqBase
from mcbo.models import ModelBase


class PI(SingleObjAcqBase):
    """
    Probability of Improvement: Prob[f(x) < f(x*)]
    """

    @property
    def name(self) -> str:
        return "PI"

    def __init__(self):
        super(PI, self).__init__()

    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 best_y: Union[float, torch.Tensor],
                 **kwargs
                 ) -> torch.Tensor:
        best_y = best_y.to(model.device, model.dtype)
        mean, var = model.predict(x)
        mean = mean.flatten()
        std = var.clamp_min(1e-9).sqrt().flatten()

        u = (best_y - mean) / std

        normal = Normal(torch.zeros(1).to(model.device), torch.ones(1).to(model.device))
        proba_of_improvement = normal.cdf(u)

        return -proba_of_improvement
