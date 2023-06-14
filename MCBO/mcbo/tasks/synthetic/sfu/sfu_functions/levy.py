# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Optional

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Levy(SfuFunction):
    """
    The Levy Function. See https://www.sfu.ca/~ssurjano/levy.html for details.
    """

    @property
    def name(self) -> str:
        return f'Levy Function{self.task_name_suffix}'

    def __init__(self, num_dims: int, lb: int = -10, ub: int = 10, task_name_suffix: Optional[str] = None):


        super(Levy, self).__init__(num_dims=num_dims, lb=lb, ub=ub)

        if task_name_suffix is None:
            task_name_suffix = ""
        self.task_name_suffix = task_name_suffix


        self.x_star = pd.DataFrame([[1 for _ in range(num_dims)]])  # Global minimiser
        if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
            self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        w = 1 + (x - 1.0) / 4.0

        y = np.sin(np.pi * w[:, 0]) ** 2 + \
            np.sum(
                (w[:, 1:self.num_dims - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, 1:self.num_dims - 1] + 1) ** 2),
                axis=1) + \
            (w[:, self.num_dims - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, self.num_dims - 1]) ** 2)

        return y.reshape(-1, 1)

