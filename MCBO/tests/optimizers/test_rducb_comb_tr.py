if __name__ == '__main__':
    import os
    from pathlib import Path
    from mcbo.optimizers.bo_builder import BoBuilder
    from mcbo.utils.plotting_utils import plot_convergence_curve
    from mcbo.task_factory import task_factory

    task = task_factory('pest')
    search_space = task.get_search_space()

    optimizer = BoBuilder(model_id="gp_rd", acq_opt_id="mp", acq_func_id="addlcb", tr_id="basic").build_bo(search_space=search_space, n_init=1, input_constraints=None)

    for i in range(200):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {optimizer.best_y[0]:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)