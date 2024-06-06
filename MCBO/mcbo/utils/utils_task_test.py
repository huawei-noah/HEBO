import time

from mcbo.optimizers import RandomSearch
from mcbo.tasks import TaskBase


def test_task(task: TaskBase, batch_size: int = 4, n_evals: int = 10):
    obj_dims = [0]
    out_constr_dims = None
    out_upper_constr_vals = None
    optimizer = RandomSearch(
        search_space=task.get_search_space(),
        input_constraints=task.input_constraints,
        obj_dims=obj_dims,
        out_constr_dims=out_constr_dims,
        out_upper_constr_vals=out_upper_constr_vals,
        store_observations=False,
    )

    print(f"{optimizer.name}_{task.name}")

    for i in range(n_evals):
        x_next = optimizer.suggest(batch_size)
        ref_time = time.time()
        y_next = task(x_next)
        print(f'Iteration {i + 1:>4d} - BBOX evaluation took {time.time() - ref_time:.2f}s')
        optimizer.observe(x_next, y_next)
        if len(obj_dims) == 1:
            print(f'Iteration {i + 1:>4d} - Best f(x) {optimizer.best_y[obj_dims].item():.3f}')
        else:
            print(f'Iteration {i + 1:>4d} - Best f(x) {optimizer.best_y}')
