# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Rastrigin(SfuFunction):
    """
    The Rastrigin Function. See https://www.sfu.ca/~ssurjano/rastr.html for details.
    """

    @property
    def name(self) -> str:
        return 'Rastrigin Function'

    def __init__(self, num_dims: int, lb: float = -5.12, ub: float = 5.12):


        super(Rastrigin, self).__init__(num_dims=num_dims, lb=lb, ub=ub)


        self.x_star = pd.DataFrame([[0 for _ in range(num_dims)]])  # Global minimiser
        if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
            self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        return (10 * self.num_dims + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum(axis=-1)).reshape(-1, 1)
