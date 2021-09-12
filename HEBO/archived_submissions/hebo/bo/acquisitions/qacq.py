# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from ..models.base_model import BaseModel
from .acq import SingleObjectiveAcq

class QEI_MC(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, tau : float, **conf):
        super().__init__(model, **conf)
        self.q   = conf.get('q', 1)
        self.tau = tau

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        # XXX: not to be used for population-based optimisation method
        assert(x.dim() == 2)
        assert(x.shape[0] == self.q)
        sample = self.model.sample_y(x, xe, n_sample = 20)
        best_y = sample.min(dim = 1).values
        return (self.tau - best_y).clamp(min = 0.).mean()

class QSR_MC(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, tau : float, **conf):
        super().__init__(model, **conf)
        self.q   = conf.get('q', 1)
        self.tau = tau

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        # XXX: not to be used for population-based optimisation method
        assert(x.dim() == 2)
        assert(x.shape[0] == self.q)
        sample = self.model.sample_y(x, xe, n_sample = 20)
        best_y = sample.min(dim = 1).values
        return best_y.mean()
