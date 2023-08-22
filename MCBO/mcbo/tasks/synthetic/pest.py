# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from mcbo.tasks import TaskBase


# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


def spread_pests(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(init_pest_frac_alpha, init_pest_frac_beta,
                                                          size=(n_simulations,))
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(control_alpha, control_beta[x[i]],
                                                                size=(n_simulations,))
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = spread_pests(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = spread_pests(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(TaskBase):
    """
	Pest Control Problem.
	"""
    categories = ['do nothing', 'pesticide 1', 'pesticide 2', 'pesticide 3', 'pesticide 4']

    @property
    def name(self) -> str:
        return 'Pest Control'

    def __init__(self, random_seed=0):
        super(PestControl, self).__init__()
        self.seed = random_seed
        self._n_choices = 5
        self._n_stages = 25

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        x_ = x.replace(['do nothing', 'pesticide 1', 'pesticide 2', 'pesticide 3', 'pesticide 4'], [0, 1, 2, 3, 4])
        x_ = x_.to_numpy()
        return np.array([self._compute(i) for i in x_])

    def _compute(self, x):
        assert x.ndim == 1 and len(x) == self._n_stages, (x.shape, self._n_stages)
        evaluation = np.array([_pest_control_score(x, seed=self.seed)])
        return evaluation

    @staticmethod
    def get_static_search_space_params(n_stages: int) -> List[Dict[str, Any]]:
        params = []
        for i in range(1, n_stages + 1):
            params.append({'name': f'stage_{i}', 'type': 'nominal', 'categories': PestControl.categories})
        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(n_stages=self._n_stages)
