# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Powell(SfuFunction):
    """
    The Powell Function. See https://www.sfu.ca/~ssurjano/permdb.html for details.
    """

    @property
    def name(self) -> str:
        return 'Powell Function'

    def __init__(self, num_dims: int, lb: float = -10, ub: float = 10):
        assert isinstance(num_dims, int) and num_dims >= 4
        assert isinstance(lb, int) or isinstance(lb, float)
        assert isinstance(ub, int) or isinstance(ub, float)

        super(Powell, self).__init__(num_dims=num_dims, lb=lb, ub=ub)



    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        sum = np.zeros(len(x))

        for i in range(1, int(self.num_dims / 4) + 1):
            term1 = (x[:, 4 * i - 4] + 10 * x[:, 4 * i - 3]) ** 2
            term2 = 5 * (x[:, 4 * i - 2] - x[:, 4 * i - 1]) ** 2
            term3 = (x[:, 4 * i - 3] - 2 * x[:, 4 * i - 2]) ** 4
            term4 = 10 * (x[:, 4 * i - 4] - x[:, 4 * i - 1]) ** 4
            sum += term1 + term2 + term3 + term4

        return sum.reshape(-1, 1)
