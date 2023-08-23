import time
from typing import Tuple, Optional

import numpy as np
import scipy
import torch
from tqdm import tqdm

from mcbo.optimizers import BoBase
from mcbo.tasks.task_base import TaskBase
from mcbo.utils.general_utils import time_formatter


def estimate_runtime(optimizer: BoBase, task: TaskBase, one_eval_runtime: Optional[float] = None,
                     total_budget: int = 200,
                     interpolation_num_samples: int = 5) -> Tuple[float, float, float, float]:
    """
    Estimate how long would a BO run take by simulating a few BO steps

    Args:
        optimizer: optimizer to test.
        task: optimization task.
        one_eval_runtime: average runtime of a single
                          black-box evaluation (if it is not known, the black-box will be queried once)
        total_budget: estimate runtime for this number of BO steps.
        interpolation_num_samples: number of simulated BO steps to do.

    Returns:
        Estimated BO step runtime
        Estimated BO step runtime + black-box evaluation runtime
        Time it took to run estimation of BO steps
        Time it took to run estimation of black-box evaluation
    """

    # --- Estimate time required to evaluate the black-box function
    init_points = optimizer.suggest(n_suggestions=optimizer.n_init)
    sample = init_points.iloc[[0]]
    if one_eval_runtime is None:
        time_ref = time.time()
        _ = task(sample)
        time_taken_to_eval_bb = time.time() - time_ref
    else:
        time_taken_to_eval_bb = one_eval_runtime
    total_time_to_eval_bb = time_taken_to_eval_bb * total_budget

    if total_budget <= optimizer.n_init:
        print(
            '\nTotal budget < initial population size. All points will be randomly sampled. Estimated runtime: <1s.\n')
        return 1, 1 + total_time_to_eval_bb, 0, time_taken_to_eval_bb

    # --- Estimate time required to train the surrogate and optimize the acquisition during each BO iteration

    search_space = optimizer.search_space

    # Find num of samples for which to evaluate how long it takes
    eval_runtime_at = np.round(np.linspace(optimizer.n_init + 1, total_budget, interpolation_num_samples)).astype(int)

    y = torch.randn(len(init_points), dtype=optimizer.dtype).unsqueeze(-1).numpy()
    optimizer.observe(x=init_points, y=y)

    # Evaluate the runtime
    runtimes = []
    for dataset_size in tqdm(eval_runtime_at):
        # Measure time take to suggest new point
        time_ref = time.time()

        optimizer.suggest(n_suggestions=1)

        # Sample random samples from the search space
        n_new_points = dataset_size - len(optimizer.data_buffer)
        x = search_space.sample(n_new_points)
        y = torch.randn(n_new_points, dtype=optimizer.dtype).unsqueeze(-1).numpy()
        optimizer.observe(x, y)

        time_taken_to_suggest_point = time.time() - time_ref
        runtimes.append(time_taken_to_suggest_point)

    # Fit a linear interpolation function to this data
    f = scipy.interpolate.interp1d(eval_runtime_at, runtimes, kind='linear')

    # Estimate runtime for all points (ignoring the first n_init points as we don't run a BO loop for them)
    num_samples = np.arange(optimizer.n_init + 1, total_budget + 1)
    expected_runtime = f(num_samples)
    total_time_to_run_bo_loops = expected_runtime.sum()

    total_runtime = total_time_to_eval_bb + total_time_to_run_bo_loops
    print(f'\nExpected BO step runtime: {time_formatter(total_time_to_run_bo_loops)}')
    print(f'Expected total runtime: {time_formatter(total_runtime)}')
    print(f'BO step estimation took: {time_formatter(sum(runtimes))}')
    print(f'Black-box runtime estimation took: {time_formatter(time_taken_to_eval_bb)}\n')
    return total_time_to_run_bo_loops, total_runtime, sum(runtimes), time_taken_to_eval_bb
