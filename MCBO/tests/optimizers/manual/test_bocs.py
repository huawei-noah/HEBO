import os
import sys
from pathlib import Path
from typing import Callable, Dict

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import torch

from mcbo.optimizers.manual.bocs import BOCS
from mcbo.utils.plotting_utils import plot_convergence_curve


def input_constraint_maker(ind: int) -> Callable[[Dict], bool]:
    def f(x: Dict) -> bool:
        return x[f"var_{ind}"] < 0

    return f


if __name__ == '__main__':
    from mcbo.task_factory import task_factory

    task = task_factory('levy', num_dims=10, variable_type='nominal', num_categories=21)
    search_space = task.get_search_space()

    input_constraints = [input_constraint_maker(i) for i in range(1, 4)]

    optimizer = BOCS(
        search_space=search_space,
        input_constraints=input_constraints,
        n_init=10,
        device=torch.device('cuda:0')
    )

    for i in range(200):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
