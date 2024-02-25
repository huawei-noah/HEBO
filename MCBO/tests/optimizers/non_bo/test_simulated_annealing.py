import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.tasks import TaskBase
from mcbo.trust_region.random_restart_tr_manager import RandomRestartTrManager

import torch

from mcbo.optimizers.non_bo.simulated_annealing import SimulatedAnnealing


def get_task_sp(mixed: bool) -> TaskBase:
    from mcbo.task_factory import task_factory
    task_name = "ackley"
    if mixed:
        num_dims = [50, 3]
        variable_type = ['nominal', 'num']
        num_categories = [2, None]
        task_name_suffix = " 50-nom-2 3-num"
        task_kwargs = dict(num_dims=num_dims, variable_type=variable_type, num_categories=num_categories,
                           task_name_suffix=task_name_suffix, lb=-1, ub=1)
        return task_factory(task_name, **task_kwargs)
    else:
        return task_factory(task_name=task_name, dtype=torch.float64, num_dims=10, variable_type='nominal',
                            num_categories=21)


def test_sa(mixed: bool, use_tr: bool):
    task = get_task_sp(mixed=mixed)
    search_space = task.get_search_space()

    tr_manager = None
    if use_tr:
        tr_manager = RandomRestartTrManager(
            search_space=search_space,
            obj_dims=[0],
            out_constr_dims=None,
            out_upper_constr_vals=None,
            min_num_radius=2 ** -5,
            max_num_radius=1.,
            init_num_radius=0.8,
            min_nominal_radius=1,
            max_nominal_radius=10,
            init_nominal_radius=8,
            fail_tol=5,
            succ_tol=2,
            verbose=True
        )
        center = search_space.transform(search_space.sample(1))[0]
        tr_manager.set_center(center)
        tr_manager.radii['nominal'] = 10

    optimizer = SimulatedAnnealing(
        search_space=search_space,
        input_constraints=task.input_constraints,
        obj_dims=[0],
        out_constr_dims=None,
        out_upper_constr_vals=None,
        fixed_tr_manager=tr_manager,
        allow_repeating_suggestions=False
    )

    for i in range(100):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) {optimizer.best_y:.3f}')


if __name__ == '__main__':
    mixed_ = True
    use_tr_ = True
    test_sa(mixed=mixed_, use_tr=use_tr_)
