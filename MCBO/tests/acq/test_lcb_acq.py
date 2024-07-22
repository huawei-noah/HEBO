import torch
from gpytorch.kernels import ScaleKernel, MaternKernel

from mcbo.task_factory import task_factory
from mcbo.models.gp.exact_gp import ExactGPModel
from mcbo.acq_funcs.factory import acq_factory

if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device('cpu')
    task = task_factory('sphere', num_dims=10, variable_type='num')
    search_space = task.get_search_space(dtype=dtype)

    x_train_pd = search_space.sample(1000)
    x_train = search_space.transform(x_train_pd)
    y_train = torch.tensor(task(x_train_pd))

    x_test_pd = search_space.sample(200)
    x_test = search_space.transform(x_test_pd)
    y_test = torch.tensor(task(x_test_pd))

    kernel = ScaleKernel(MaternKernel(ard_num_dims=search_space.num_dims))

    model = ExactGPModel(search_space, 1, kernel)

    model.fit(x_train, y_train)

    acq_func = acq_factory('lcb')

    acq_func(x=x_test, model=model, best_y=y_train.min())