# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import sys
from pathlib import Path

import torch

from mcbo.optimizers.bo_builder import BoBuilder

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.utils.plotting_utils import plot_convergence_curve
from mcbo.task_factory import task_factory
from mcbo.utils.distance_metrics import hamming_distance

if __name__ == '__main__':
    use_tr = True
    n_init = 20
    dtype = torch.float64

    task = task_factory('levy', num_dims=[3, 3, 6], variable_type=['nominal', 'num', 'nominal'],
                        num_categories=[5, 5, 5])
    search_space = task.get_search_space(dtype=dtype)

    cocabo_builder = BoBuilder(model_id="gp_o", acq_opt_id="mab", acq_func_id="ei", tr_id="basic")
    optimizer = cocabo_builder.build_bo(search_space=search_space, n_init=n_init)

    for i in range(100):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        if use_tr and optimizer.tr_manager.center is not None and i >= n_init:
            nominal_dist = hamming_distance(search_space.transform(x_next)[0, search_space.nominal_dims],
                                            optimizer.tr_manager.center[search_space.nominal_dims], normalize=False)
            if search_space.num_numeric > 0 and search_space.num_nominal > 0:
                max_numeric_dist = torch.abs(
                    search_space.transform(x_next)[0, search_space.cont_dims + search_space.disc_dims] -
                    optimizer.tr_manager.center[search_space.cont_dims + search_space.disc_dims]).max().item()
                print(
                    f'Iteration {i + 1:>4d} - f(x) {y_next[0, 0]:.3f} - f(x*) {optimizer.best_y:.3f} - Suggestion within tr: {nominal_dist <= optimizer.tr_manager.radii["nominal"] and max_numeric_dist < optimizer.tr_manager.radii["numeric"]}')
            elif search_space.num_numeric > 0:
                max_numeric_dist = torch.abs(
                    search_space.transform(x_next)[0, search_space.cont_dims + search_space.disc_dims] -
                    optimizer.tr_manager.center[search_space.cont_dims + search_space.disc_dims]).max().item()
                print(
                    f'Iteration {i + 1:>4d} - f(x) {y_next[0, 0]:.3f} - f(x*) {optimizer.best_y:.3f} - Suggestion within tr: {max_numeric_dist < optimizer.tr_manager.radii["numeric"]}')
            elif search_space.num_nominal > 0:
                print(
                    f'Iteration {i + 1:>4d} - f(x) {y_next[0, 0]:.3f} - f(x*) {optimizer.best_y:.3f} - Suggestion within tr: {nominal_dist <= optimizer.tr_manager.radii["nominal"]}')

        else:
            print(f'Iteration {i + 1:>4d} - f(x) {y_next[0, 0]:.3f} - f(x*) {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
