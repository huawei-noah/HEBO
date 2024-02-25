import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel

from mcbo.task_factory import task_factory
from mcbo.models.gp.exact_gp import ExactGPModel

if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device('cpu')
    task1 = task_factory('levy', num_dims=10, variable_type='num')
    task2 = task_factory('sphere', num_dims=10, variable_type='num')
    search_space = task1.get_search_space(dtype=dtype)

    x_train_pd = search_space.sample(1000)
    x_train = search_space.transform(x_train_pd)
    y1_train_np = task1(x_train_pd)
    y2_train_np = task2(x_train_pd)
    y_train = torch.tensor(np.hstack((y1_train_np, y2_train_np)))

    x_test_pd = search_space.sample(200)
    x_test = search_space.transform(x_test_pd)
    y1_test_np = task2(x_test_pd)
    y2_test_np = task2(x_test_pd)
    y_test = torch.tensor(np.hstack((y1_test_np, y2_test_np)))

    kernel = ScaleKernel(MaternKernel(ard_num_dims=search_space.num_dims))

    model = ExactGPModel(search_space, 2, kernel)

    model.fit(x_train, y_train)

    y_pred, var_pred = model.predict(x_test)

    # Plotting should be improved by ordering according to the true y values and by using two seperate plots.
    # plot_posterior(y_pred[:, 0].detach().cpu().numpy(), var_pred[:, 0].sqrt().detach().cpu().numpy(),
    #                y_test[:, 0].numpy(), './', save=True)
    #
    # plot_posterior(y_pred[:, 1].detach().cpu().numpy(), var_pred[:, 1].sqrt().detach().cpu().numpy(),
    #                y_test[:, 1].numpy(), './', save=True)
