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


class EI(SingleObjAcqBase):
    """
    Expected Improvement
    """

    @property
    def name(self) -> str:
        if self.augmented_ei:
            return "aug-EI"
        return "EI"

    def __init__(self, augmented_ei: bool = False):
        super(EI, self).__init__()
        self.augmented_ei = augmented_ei

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

        # # use in-fill criterion
        # mu_star, _ = self.model.predict(self.x_best.view(1, -1))

        u = (best_y - mean) / std

        normal = Normal(torch.zeros(1).to(model.device), torch.ones(1).to(model.device))
        ucdf = normal.cdf(u)

        updf = torch.exp(normal.log_prob(u))
        ei = (std * (u * ucdf + updf))
        if self.augmented_ei:
            # Only difference from normal ei
            sigma_n = model.noise
            ei = ei * (1. - sigma_n.clamp_min(1e-9).sqrt() / torch.sqrt(sigma_n + std ** 2))

        # We return the negative of the expected improvement as we are minimizing the acquisition function
        return -ei
