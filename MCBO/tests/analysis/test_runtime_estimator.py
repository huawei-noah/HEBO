import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import time

import numpy as np
import torch

from analysis.runtime_estimator import estimate_runtime
from mcbo.utils.general_utils import time_formatter


def test_runtime_estimator(algo_name: str) -> int:
    # Initialise a task
    task_name = 'ackley'
    if algo_name in ["CoCaBO"]:
        task = task_factory(task_name=task_name, num_dims=[2, 2, 2, 2, 2],
                            variable_type=['nominal', 'int', 'nominal', 'int', 'nominal'],
                            num_categories=[15, None, 15, None, 15])
    else:
        task = task_factory(task_name=task_name, variable_type=['nominal'], num_dims=[10], num_categories=[15])
    search_space = task.get_search_space()

    # Get an algorithm
    bo_builder = BO_ALGOS[algo_name]

    optimizer = bo_builder.build_bo(search_space=search_space, n_init=20, device=torch.device("cuda"))

    total_budget = 200

    # Measure actual runtime
    time_ref = time.time()
    # Get actual runtime
    for i in range(total_budget):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i + 1:3d}/{total_budget:3d} - f(x) = {y[0, 0]:.3f} - f(x*) = {optimizer.best_y:.3f}')

    actual_time_taken = int(np.round(time.time() - time_ref))

    optimizer = bo_builder.build_bo(search_space=search_space, n_init=20, device=torch.device("cuda"))

    interpolation_num_samples = 5
    total_budget = 200

    # Estimate runtime
    estimate_runtime(
        optimizer=optimizer,
        task=task,
        total_budget=total_budget,
        interpolation_num_samples=interpolation_num_samples
    )

    # Compare to actual runtime
    print(f'\nActual Runtime: {time_formatter(actual_time_taken)}\n')
    return 0


if __name__ == '__main__':
    from mcbo.task_factory import task_factory
    from mcbo.optimizers.bo_builder import BO_ALGOS

    for bo_algo in ["BODi", "COMBO", "BOiLS", "BOSS", "Casmopolitan", "BOCS", "CoCaBO"]:
        print(bo_algo)
        test_runtime_estimator(algo_name=bo_algo)
