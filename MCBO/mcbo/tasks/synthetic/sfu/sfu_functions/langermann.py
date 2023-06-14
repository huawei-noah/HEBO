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


class Langermann(SfuFunction):
    """
    The Langermann Function. See https://www.sfu.ca/~ssurjano/langer.html for details.
    """

    @property
    def name(self) -> str:
        return 'Langermann Function'

    def __init__(self, num_dims: int, lb: float = -0., ub: float = 10., m: Optional[int] = None,
                 c: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None):


        super(Langermann, self).__init__(num_dims=num_dims, lb=lb, ub=ub)



        # Default case when m == 5 and num_dims == 2
        if (m is None or m == 5) and num_dims == 2:
            self.m = 5
            if c is None:
                self.c = np.array([[1., 2., 5., 2., 3.]])
            else:
                assert isinstance(c, np.ndarray)
                assert c.ndim == 2
                assert c.shape[0] == 1
                assert c.shape[1] == self.m
                self.c = c.astype(float)

            if a is None:
                self.a = np.array([[3., 5.], [5., 2.], [2., 1.], [1., 4], [7., 9]])
            else:
                assert isinstance(a, np.ndarray)
                assert a.ndim == 2
                assert a.shape[0] == self.m
                assert a.shape[1] == self.num_dims
                self.a = a.astype(float)
        else:
            if m is None:
                warnings.warn('Value of m was not specified. Setting m = 5.')
                self.m = 5
            else:
                assert isinstance(m, int)
                self.m = m

            if c is None:
                warnings.warn(
                    'Elements of c not specified. Initialising vector c with random integers from the interval [1, 5]')
                self.c = np.random.randint(low=1, high=5, size=(1, self.m)).astype(float)

            else:
                assert isinstance(c, np.ndarray)
                assert c.ndim == 2
                assert c.shape[0] == 1
                assert c.shape[1] == self.m
                self.c = c.astype(float)

            if a is None:
                warnings.warn(
                    'Elements of a not specified. Initialising matrix a with random integers from the interval [1, 10]')
                self.a = np.random.randint(low=1, high=10, size=(self.m, self.num_dims)).astype(float)
            else:
                assert isinstance(a, np.ndarray)
                assert a.ndim == 2
                assert a.shape[0] == self.m
                assert a.shape[1] == self.num_dims
                self.a = a.astype(float)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        temp = ((np.expand_dims(x, 1) - np.expand_dims(self.a, 0)) ** 2).sum(axis=-1)

        return (self.c * np.exp(- temp / np.pi) * np.cos(np.pi * temp)).sum(axis=-1).reshape(-1, 1)
