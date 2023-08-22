import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mcbo.models import ExactGPModel
from mcbo.models.gp.kernel_factory import kernel_factory
from mcbo.search_space import SearchSpace
from mcbo.tasks import TaskBase


class Sin(TaskBase):
    @property
    def name(self) -> str:
        return 'Sin Function'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        return np.sin(x.to_numpy())


if __name__ == '__main__':
    dtype = torch.float64
    search_space = SearchSpace([{'name': 'x', 'type': 'num', 'lb': 0, 'ub': 2 * np.pi}], dtype)
    task = Sin()

    x_train = pd.DataFrame(np.array([[0.1],
                                     [1.2],
                                     [1.4],
                                     [1.6],
                                     # [2.9],
                                     [4.],
                                     [5.2]]), columns=['x'])
    y_train = task(x_train)

    kernel = kernel_factory('mat52', active_dims=search_space.cont_dims, )
    gp = ExactGPModel(search_space, 1, kernel, num_epochs=2000)

    gp.fit(search_space.transform(x_train), torch.Tensor(y_train))

    x_test = np.linspace(0, 2 * np.pi, 100)
    x_test = pd.DataFrame(x_test, columns=['x'])
    y_test = task(x_test)

    with torch.no_grad():
        mu, var = gp.predict(search_space.transform(x_test))

    # flatten everything
    x_train = x_train.to_numpy().flatten()
    y_train = y_train.flatten()

    x_test = x_test.to_numpy().flatten()
    y_test = y_test.flatten()

    mu = mu.flatten().cpu().numpy()
    std = var.sqrt().flatten().cpu().numpy()

    # Make a plot of GP fit on training set
    plt.figure()
    plt.ylabel('Model Prediction')
    plt.xlabel('x')
    plt.grid()
    plt.xlim(0, 2*np.pi)
    plt.plot(x_train, y_train, 'ok')
    # plt.plot(x_test, y_test, '--k', label='Ground truth')
    plt.plot(x_test, mu, '-b', label='mean')
    plt.fill_between(x_test, mu - std, mu + std, color='b', alpha=0.2)
    plt.savefig('./gp_fit.png', dpi=480, transparent=True)
    plt.close()

    # linear regression
    X_train = np.concatenate((np.ones((len(x_train), 1)), x_train.reshape(-1, 1)), axis=1)
    Y_train = y_train.reshape(-1, 1)
    W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train

    lin_pred = np.concatenate((np.ones((len(x_test), 1)), x_test.reshape(-1, 1)), axis=1) @ W
    lin_pred = lin_pred.flatten()

    # Make a plot of GP fit on training set
    plt.figure()
    plt.ylabel('Model Prediction')
    plt.xlabel('x')
    plt.grid()
    plt.xlim(0, 2*np.pi)
    plt.plot(x_train, y_train, 'ok')
    # plt.plot(x_test, y_test, '--k', label='Ground truth')
    plt.plot(x_test, lin_pred, '-b', label='mean')
    plt.savefig('./linear_fit.png', dpi=480, transparent=True)
    plt.close()