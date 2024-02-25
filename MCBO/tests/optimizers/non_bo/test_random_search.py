import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.task_factory import task_factory
from mcbo.optimizers import RandomSearch
from mcbo.utils.general_utils import set_random_seed
from mcbo.utils.plotting_utils import plot_convergence_curve

if __name__ == '__main__':
    set_random_seed(2)

    task = task_factory('levy', num_dims=10, variable_type='nominal', num_categories=21)
    search_space = task.get_search_space()

    optimizer = RandomSearch(search_space, store_observations=True)

    x_init = search_space.sample(2)
    optimizer.set_x_init(x_init)

    for i in range(200):
        x_next = optimizer.suggest(2)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
