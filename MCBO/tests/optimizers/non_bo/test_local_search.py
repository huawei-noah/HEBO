import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT


if __name__ == '__main__':
    from mcbo.optimizers.non_bo.hill_climbing import HillClimbing
    from mcbo.utils.plotting_utils import plot_convergence_curve
    from mcbo.task_factory import task_factory

    task = task_factory('levy', num_dims=10, variable_type='nominal', num_categories=21)
    search_space = task.get_search_space()

    optimizer = HillClimbing(
        search_space=search_space,
        input_constraints=task.input_constraints,
        obj_dims=[0],
        out_upper_constr_vals=None,
        out_constr_dims=None
    )

    for i in range(500):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
