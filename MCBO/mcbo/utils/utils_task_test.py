from mcbo.optimizers import RandomSearch
from mcbo.tasks import TaskBase


def test_task(task: TaskBase):
    optimizer = RandomSearch(
        search_space=task.get_search_space(),
        input_constraints=task.input_constraints,
        store_observations=False
    )

    print(f"{optimizer.name}_{task.name}")

    for i in range(5):
        x_next = optimizer.suggest(4)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - Best f(x) {optimizer.best_y:.3f}')
