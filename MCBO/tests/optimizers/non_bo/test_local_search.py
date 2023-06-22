import os
from pathlib import Path
import sys

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

import torch

from mcbo.optimizers.non_bo.local_search import LocalSearch
from mcbo.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    from mcbo.task_factory import task_factory

    task = task_factory('levy', num_dims=10, variable_type='nominal', num_categories=21)
    search_space = task.get_search_space()

    optimizer = LocalSearch(search_space, input_constraints=task.input_constraints)

    for i in range(500):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
