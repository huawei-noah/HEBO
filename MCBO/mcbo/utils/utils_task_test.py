import time

from mcbo.optimizers import RandomSearch
from mcbo.tasks import TaskBase


def test_task(task: TaskBase, batch_size: int = 4, n_evals: int = 10):
    optimizer = RandomSearch(
        search_space=task.get_search_space(),
        input_constraints=task.input_constraints,
        store_observations=False
    )

    print(f"{optimizer.name}_{task.name}")

    for i in range(n_evals):
        x_next = optimizer.suggest(batch_size)
        ref_time = time.time()
        y_next = task(x_next)
        print(f'Iteration {i + 1:>4d} - BBOX evaluation took {time.time() - ref_time:.2f}s')
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - Best f(x) {optimizer.best_y:.3f}')
