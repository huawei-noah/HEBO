# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional, List

import torch

from mcbo.search_space import SearchSpace


class ModelBase(ABC):
    supports_cuda = False
    support_ts = False
    support_grad = False
    support_multi_output = False
    support_warm_start = False
    ensemble = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __init__(self, search_space: SearchSpace, num_out: int, dtype: torch.dtype, device: torch.device, **kwargs):
        """
        Base class for probabilistic regression models
        """
        super(ModelBase, self).__init__()

        self.x = None
        self._y = None
        self.fit_y = None  # model should fit on (self.x, self.fit_y) -> self.fit_y typically standardized self._y

        self.num_out = num_out
        self.search_space = search_space
        self.cont_dims = search_space.cont_dims
        self.disc_dims = search_space.disc_dims
        self.ordinal_dims = search_space.ordinal_dims
        self.nominal_dims = search_space.nominal_dims
        self.perm_dims = search_space.perm_dims
        self.dtype = dtype
        self.device = device
        self.kwargs = kwargs

        # Basic checks
        assert self.num_out > 0
        assert (len(self.cont_dims) >= 0)
        assert (len(self.disc_dims) >= 0)
        assert (len(self.ordinal_dims) >= 0)
        assert (len(self.nominal_dims) >= 0)
        assert (len(self.perm_dims) >= 0)
        assert (len(self.cont_dims) + len(self.disc_dims) + len(self.ordinal_dims) + len(self.nominal_dims) + len(
            self.perm_dims) > 0)

    @property
    def y_mean(self) -> torch.Tensor:
        if self._y is not None and len(self._y) > 0:
            return self._y.mean(axis=0)
        return torch.tensor(0, dtype=self.dtype, device=self.device)

    @property
    def y_std(self) -> torch.Tensor:
        if self._y is not None and len(self._y) > 1:
            std = self._y.std(axis=0)
            std[std < 1e-6] = 1
            return std
        return torch.tensor(1.0, dtype=self.dtype, device=self.device)

    @abstractmethod
    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        """ Transform raw ys to normalized / standardized ys
        Args:
            y: tensor of shape (n, num_outputs)
        """
        pass

    @abstractmethod
    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        """ Go from normalized / standardized fit_ys to original ys
        Args:
            fit_y: tensor of shape (n, num_outputs)
        """
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        """
        Function used to fit the parameters of the model

        Args:
            x: points in transformed search space
            y: values
            kwargs: optional keyword arguments

        Returns:
            (Optional) a list of stats
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Function used to return the mean and variance of the (possibly approximated) Gaussian predictive distribution
        for the input x. Output shape of each tensor is (N, num_out) where N is the number of input points

        Args:
            x: points in transformed search space
            kwargs: optional keyword arguments

        Returns:
            predicted mean and variance
        """
        pass

    @property
    @abstractmethod
    def noise(self) -> torch.Tensor:
        """
        Return estimated noise variance, for example, GP can view noise level as a hyperparameter and optimize it via
        MLE, another strategy could be using the MSE of training data as noise estimation Should return a float tensor
        of shape self.num_out

        Returns:
            estimated noise variance
        """
        pass

    def sample_y(self, x: torch.Tensor, n_samples: int, **kwargs) -> torch.Tensor:
        # TODO: add option to take into account the covariance
        py, ps2 = self.predict(x)
        ps = ps2.sqrt()
        samp = torch.zeros(n_samples, py.shape[0], self.num_out)
        for i in range(n_samples):
            samp[i] = py + ps * torch.randn(py.shape).to(py)
        return samp

    @abstractmethod
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> ModelBase:
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        Args:
            device: target device
            dtype: target dtype

        Returns:
            self
        """
        pass

    def pre_fit_method(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        """
        Function called before fitting the model in the suggest method. Can be used to update the internal state
        of the model based on the data that will be used to fit the model. Use cases may include training a VAE
        for latent space BO, or re-initialising the model before fitting it to the data.

        Args:
            x: points in transformed space
            y: values
            kwargs: optional keyword arguments
        :return:
        """
        pass


class EnsembleModelBase(ModelBase, ABC):
    """
    Ensemble of models. This class is commonly used when sampling from the model's parameter's posterior.
    """

    ensemble = True

    def __init__(self,
                 search_space: SearchSpace,
                 num_out: int,
                 num_models: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 **kwargs):
        self.num_models = num_models
        self.models = []

        super(EnsembleModelBase, self).__init__(search_space, num_out, dtype, device)

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        """
        Function used to fit num_models models

        Args:
            x: points in transformed space
            y: values
            kwargs: optional keyword arguments
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Function used to return the mean and variance of the (possibly approximated) Gaussian predictive distribution
        for the input x. Output shape (N, num_out, num_models) where N is the number of input points.

        If the model uses a device, this method should automatically move x to the target device.

        Args:
            x: points in transformed space
            kwargs: optional keyword arguments
        :return:
        """
        pass

    @property
    def noise(self) -> torch.Tensor:
        """
        Return estimated noise variance, for example, GP can view noise level
        as a hyperparameter and optimize it via MLE, another strategy could be
        using the MSE of training data as noise estimation
        Should return a float tensor of shape self.num_out
        """
        noise = 0
        for model in self.models:
            noise += model.noise

        return noise / len(self.models)

    def sample_y(self, x: torch.Tensor, n_samples: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> ModelBase:
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        Args:
            device: target device
            dtype: target dtype

        Returns:
            self
        """
        self.models = [model.to(device=device, dtype=dtype) for model in self.models]
        return self

    def pre_fit_method(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        """
        Function called at the before fitting the model in the suggest method. Can be used to update the internal state
        of the model based on the data that will be used to fit the model. Use cases may include training a VAE
        for latent space BO, or re-initialising the model before fitting it to the data.

        Args:
            x: points in transformed space
            y: values
            kwargs: optional keyword arguments
        """
        for model in self.models:
            model.pre_fit_method(x, y, **kwargs)
