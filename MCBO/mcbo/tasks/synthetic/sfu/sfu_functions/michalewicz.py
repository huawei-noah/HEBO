# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Michalewicz(SfuFunction):
    """
    The Michalewicz Function. See https://www.sfu.ca/~ssurjano/michal.html for details.
    """

    @property
    def name(self) -> str:
        return 'Michalewicz Function'

    def __init__(self, num_dims: int, lb: float = -5., ub: float = 10., m: float = 10.):

        assert isinstance(m, int) or isinstance(m, float)

        super(Michalewicz, self).__init__(num_dims=num_dims, lb=lb, ub=ub)


        self.m = m
        self.indices = np.arange(1, self.num_dims + 1).reshape(1, -1)

        if num_dims == 2:
            self.x_star = pd.DataFrame([[2.20, 1.57]])  # Global minimiser
            if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
                self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum
        elif num_dims == 5:
            self.global_optimum = -4.687658
        elif num_dims == 10:
            self.global_optimum = -9.66015

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        return - (np.sin(x) * np.sin(self.indices * x ** 2 / np.pi) ** (2 * self.m)).sum(axis=-1).reshape(-1, 1)
