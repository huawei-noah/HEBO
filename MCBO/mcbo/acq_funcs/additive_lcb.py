# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Tuple
import torch

from mcbo.acq_funcs.lcb import LCB
from mcbo.models.gp.rand_decomposition_gp import RandDecompositionGP


class AddLCB(LCB):
    def __init__(self, beta: float = 1.96):
        super().__init__(beta)
    
    def evaluate(self,
                 x: torch.Tensor,
                 model: RandDecompositionGP,
                 **kwargs
                 ) -> torch.Tensor:
        val = torch.tensor([0])
        for clique in model.graph:
            aux = self.partial_evaluate(x, model, clique, **kwargs)
            val = val.to(aux)
            val += aux

        return val
    
    def partial_evaluate(self,
                 x: torch.Tensor,
                 model: RandDecompositionGP,
                 clique: Tuple[int],
                 **kwargs
                 ) -> torch.Tensor:
        mean, var = model.partial_predict(x, clique)
        mean = mean.flatten()
        std = var.clamp_min(1e-9).sqrt().flatten()

        return mean - self.beta * std
