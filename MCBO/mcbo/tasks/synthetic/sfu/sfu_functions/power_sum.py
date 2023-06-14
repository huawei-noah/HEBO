# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class PowSum(SfuFunction):
    """
    The Power Sum Function. See https://www.sfu.ca/~ssurjano/powersum.html for details.
    Default ub = num_dims
    """

    @property
    def name(self) -> str:
        return 'Power Sum Function'

    def __init__(self, num_dims: int, lb: float = 0, ub: float = 10, b: Optional[np.ndarray] = None):


        super(PowSum, self).__init__(num_dims=num_dims, lb=lb, ub=ub)



        if b is None:
            if num_dims == 4:
                self.b = np.array([[8., 18., 44., 114.]])
            else:
                warnings.warn(
                    'Elements of b not specified. Initialising vector b with random integers from the interval [1, 100]')
                self.b = np.random.randint(low=1, high=100, size=(1, self.num_dims)).astype(float)
        else:
            assert isinstance(b, np.ndarray)
            assert b.ndim == 2
            assert b.shape[0] == 1
            assert b.shape[1] == num_dims
            self.b = b

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        x_repeat = np.repeat(np.expand_dims(x, -1), self.num_dims, -1)
        powers = np.arange(1, self.num_dims + 1).reshape(1, 1, -1) * np.ones((len(x), self.num_dims, self.num_dims))

        return (((x_repeat ** powers).sum(axis=1) - self.b) ** 2).sum(axis=-1).reshape(-1, 1)
