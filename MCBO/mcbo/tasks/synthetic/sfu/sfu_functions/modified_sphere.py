# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class ModifiedSphere(SfuFunction):
    """
    The Modified Sphere Function. See https://www.sfu.ca/~ssurjano/spheref.html for details.
    """

    @property
    def name(self) -> str:
        return 'Modified Sphere Function'

    def __init__(self, num_dims: int, lb: float = 0, ub: float = 1):


        super(ModifiedSphere, self).__init__(num_dims=num_dims, lb=lb, ub=ub)


        self.multiplier = 2 ** np.expand_dims(np.arange(1, self.num_dims + 1), 0)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        return (((x ** 2 * self.multiplier).sum(axis=-1) - 1745) / 899).reshape(-1, 1)
