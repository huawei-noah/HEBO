# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch

from mcbo.search_space.params.param import Parameter


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class SigmoidPara(Parameter):
    """
    Given specified lower range of Lb and Ub, the search is done in lb = logit(Lb), ub = logit(Ub)
    """

    def __init__(self, param_dict: dict, dtype: torch.dtype):
        super().__init__(param_dict, dtype)
        assert 0 < param_dict.get('lb') < param_dict.get('ub') < 1, 'The lower and upper bound must be greater than 0'
        self.original_ub = param_dict.get('ub')
        self.original_lb = param_dict.get('lb')
        self.lb = logit(self.original_lb)
        self.ub = logit(self.original_ub)

    def sample(self, num=1):
        assert (num > 0)
        return sigmoid(np.random.uniform(self.lb, self.ub, num))

    def transform(self, x):
        logit_x = logit(x)

        # Normalise
        normalised_logit_x = (logit_x - self.lb) / (self.ub - self.lb)

        return torch.tensor(normalised_logit_x, dtype=self.dtype)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        x = x.cpu().detach().numpy()
        # Un-normalise
        logit_x = (self.ub - self.lb) * x + self.lb

        return sigmoid(logit_x)

    @property
    def is_disc(self) -> bool:
        return False

    @property
    def is_cont(self) -> bool:
        return True

    @property
    def is_nominal(self) -> bool:
        return False

    @property
    def is_ordinal(self) -> bool:
        return False

    @property
    def is_permutation(self) -> bool:
        return False

    @property
    def is_disc_after_transform(self) -> bool:
        return False

    @property
    def opt_lb(self):
        return float(self.lb)

    @property
    def opt_ub(self):
        return float(self.ub)

    @property
    def transfo_lb(self) -> float:
        return 0

    @property
    def transfo_ub(self) -> float:
        return 1