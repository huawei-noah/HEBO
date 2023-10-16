# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch


class Parameter(ABC):
    def __init__(self, param_dict: dict, dtype: torch.dtype):
        self.name = param_dict.get('name')
        self.param_dict = param_dict
        self.dtype = dtype
        pass

    @abstractmethod
    def sample(self, num=1) -> pd.DataFrame:
        """ Sample param in original space """
        pass

    @abstractmethod
    def transform(self, x: np.array) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def is_cont(self) -> bool:
        """
        This applies to the transformed space
        """
        pass

    @property
    @abstractmethod
    def is_disc(self) -> bool:
        """
        This applies to the transformed space
        """
        pass

    @property
    @abstractmethod
    def is_nominal(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_ordinal(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_permutation(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_disc_after_transform(self) -> bool:
        pass

    @property
    @abstractmethod
    def opt_lb(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_ub(self) -> float:
        pass

    @property
    @abstractmethod
    def transfo_lb(self) -> float:
        """
        Should be such that, sampling uniformly in [transfo_lb, transfo_ub] and applying
            inverse transform should have the same distribution as sampling directly uniformly in the original space
        """
        pass

    @property
    @abstractmethod
    def transfo_ub(self) -> float:
        """
        Should be such that, sampling uniformly in [transfo_lb, transfo_ub] and applying
            inverse transform should have the same distribution as sampling directly uniformly in the original space
        """
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
