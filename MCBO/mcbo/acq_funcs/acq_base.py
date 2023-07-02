# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod
from typing import Union

import torch

from mcbo.models import ModelBase, EnsembleModelBase


class AcqBase(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_obj(self):
        pass

    @property
    @abstractmethod
    def num_constr(self):
        pass

    @abstractmethod
    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 **kwargs
                 ) -> torch.Tensor:
        """
        Function used to compute the acquisition function. Should take as input a 2D tensor with shape (N, D) where N
        is the number of data points and D is their dimensionality, and should output a 1D tensor with shape (N).

        Important: All acq_funcs are minimized. Hence, it is often necessary to return negated acquisition values.

        Args:
            x: input points in transformed space
            model: surrogate model
            kwargs: can contain best value observed so far best_y

        Returns:
            acquisition values at x
        """
        pass

    def __call__(self, x: torch.Tensor, model: Union[ModelBase, EnsembleModelBase], **kwargs) -> torch.Tensor:
        assert model.num_out == 1
        assert isinstance(model, (ModelBase, EnsembleModelBase))

        ndim = x.ndim
        dtype = model.dtype
        device = model.device

        if ndim == 1:
            x = x.view(1, -1)

        if isinstance(model, EnsembleModelBase):  # expected value of the acquisition function
            acq_values = torch.zeros((len(x), len(model.models))).to(x)

            for i, model_ in enumerate(model.models):
                acq_values[:, i] = self.evaluate(x=x, model=model_, **kwargs)
            acq = acq_values.mean(dim=1)

        else:
            acq = self.evaluate(x.to(device, dtype), model, **kwargs)

        if ndim == 1:
            acq = acq.squeeze(0)

        return acq


class SingleObjAcqBase(AcqBase):
    """
    Single-objective, unconstrained acquisition
    """

    def __init__(self, **kwargs):
        super(SingleObjAcqBase, self).__init__(**kwargs)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0
