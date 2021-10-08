import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood

from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP


def test_gp_fit(x: np.ndarray, y: np.ndarray, ax=None, title='GP fit'):
    n_held_out = int(.25 * len(y))
    indices = np.random.permutation(np.arange(len(y)))
    X_train = torch.tensor(x[indices[n_held_out:]])
    y_train = torch.tensor(np.atleast_2d(y[indices[n_held_out:]]).T)
    y_mean, y_std = y_train.mean(), y_train.std()
    X_test = torch.tensor(x[indices[:n_held_out]])
    y_test = torch.tensor(np.atleast_2d(y[indices[:n_held_out]]).T)

    model = SingleTaskGP(X_train, (y_train - y_mean) / y_std)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    if ax is None:
        ax = plt.subplot()

    stds = model(X_test).stddev.detach().numpy()
    means = model(X_test).loc.detach().numpy()
    ax.scatter(np.arange(len(X_test)), means)
    ax.errorbar(np.arange(len(X_test)), means, yerr=3 * stds, linestyle='', capthick=1,
                 label='Posterior mean (+- 3 stds)')
    true_y = ((y_test - y_mean) / y_std).detach().numpy()
    ax.scatter(np.arange(len(X_test)), true_y, label='Blackbox evaluation', alpha=.6)
    ax.legend()
    ax.set_title(title + f" ({len(y_train)} train points, {len(y_test)} test points)")
    return ax