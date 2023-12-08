# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod
from typing import Union, Optional, List

import torch

from mcbo.models import ModelBase, EnsembleModelBase


class AcqBase(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_obj(self) -> int:
        pass

    @property
    @abstractmethod
    def num_constr(self) -> int:
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

        !!! IMPORTANT: All acq_funcs are MINIMIZED. Hence, it is often necessary to return negated acquisition values.

        Args:
            x: input points in transformed space
            model: surrogate model
            kwargs: can contain best value observed so far best_y

        Returns:
            acquisition values at x
        """
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, model: Union[ModelBase, EnsembleModelBase], **kwargs) -> torch.Tensor:
        pass


class ConstrAcqBase(AcqBase, ABC):

    @abstractmethod
    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 constr_models: List[ModelBase],
                 out_upper_constr_vals: torch.Tensor,
                 **kwargs
                 ) -> torch.Tensor:
        """
        Function used to compute the acquisition function. Should take as input a 2D tensor with shape (N, D) where N
        is the number of data points and D is their dimensionality, and should output a 1D tensor with shape (N).

        Important: All acq_funcs are minimized. Hence, it is often necessary to return negated acquisition values.
        """
        pass

    def __call__(self, x: torch.Tensor, model: Union[ModelBase, EnsembleModelBase],
                 constr_models: Optional[List[ModelBase]] = None,
                 out_upper_constr_vals: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert model.num_out == 1
        assert isinstance(model, ModelBase)
        assert not model.ensemble

        if constr_models is None:
            constr_models = []
        if out_upper_constr_vals is None:
            out_upper_constr_vals = torch.zeros(0)

        ndim = x.ndim
        dtype = model.dtype
        device = model.device

        if ndim == 1:
            x = x.view(1, -1)

        if isinstance(model, EnsembleModelBase):  # expected value of the acquisition function
            acq_values = torch.zeros((len(x), len(model.models))).to(x)

            for i, model_ in enumerate(model.models):
                acq_values[:, i] = self.evaluate(
                    x=x.to(device, dtype), model=model,
                    constr_models=constr_models,
                    out_upper_constr_vals=out_upper_constr_vals,
                    **kwargs
                )
            acq = acq_values.mean(dim=1)

        else:
            acq = self.evaluate(
                x=x.to(device, dtype),
                model=model,
                constr_models=constr_models,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )

        if ndim == 1:
            acq = acq.squeeze(0)

        return acq


class SingleObjAcqBase(AcqBase, ABC):
    """
    Single-objective, unconstrained acquisition
    """

    def __init__(self, **kwargs) -> None:
        super(SingleObjAcqBase, self).__init__(**kwargs)

    @property
    def num_obj(self) -> int:
        return 1

    @property
    def num_constr(self) -> int:
        return 0

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
