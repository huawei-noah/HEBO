from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_mean_std(*args, n_std: Optional[int] = 1,
                  ax: Optional[Axes] = None, alpha: float = .3, label: Optional[str] = None,
                  **plot_mean_kwargs):
    """ Plot mean and std (with fill between) of sequentiql data Y of shape (n_trials, lenght_of_a_trial)

    Args:
        X: x-values (if None, we will take `range(0, len(Y))`)
        Y: y-values
        n_std: number of std to plot around the mean (if `0` only the mean is plotted)
        label: label to use for the mean
        ax: axis on which to plot the curves
        color: color of the curve
        alpha: parameter for `fill_between`

    Returns:
        The axis.
    """
    if len(args) == 1:
        Y = args[0]
        X = None
    elif len(args) == 2:
        X, Y = args
    else:
        raise RuntimeError('Wrong number of arguments (should be [X], Y,...)')

    assert len(Y) > 0, 'Y should be a non-empty array, nothing to plot'
    Y = np.atleast_2d(Y)
    if X is None:
        X = np.arange(Y.shape[1])
    assert X.ndim == 1, f'X should be of rank 1, got {X.ndim}'
    mean = Y.mean(0)
    std = Y.std(0)
    if ax is None:
        ax = plt.subplot()

    line_plot = ax.plot(X, mean, label=label, **plot_mean_kwargs)

    if n_std > 0 and Y.shape[0] > 1:
        ax.fill_between(X, mean - n_std * std, mean + n_std * std, alpha=alpha, color=line_plot[0].get_c())

    return ax