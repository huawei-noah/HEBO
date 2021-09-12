# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Dict, Any
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import List, Union, Tuple


def cummax(X: np.ndarray, return_ind=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Return array containing at index `i` the value max(X)[:i] """
    cmaxind: List[int] = [0]
    cmax: List[float] = [X[0]]
    for i, x in enumerate(X[1:]):
        i += 1
        if x > cmax[-1]:
            cmax.append(x)
            cmaxind.append(i)
        else:
            cmax.append(cmax[-1])
            cmaxind.append(cmaxind[-1])
    cmax_np = np.array(cmax)
    assert np.all(X[cmaxind] == cmax_np), (X, X[cmaxind], cmax_np)
    if return_ind:
        return cmax_np, np.array(cmaxind)
    return cmax_np


def get_cummax(scores: List[np.ndarray]) -> List[np.ndarray]:
    """ Compute cumulative max for each array in a list

    Args:
        scores: list of the arrays on which `cummax` will be applied

    Returns:
        cmaxs:
    """
    if isinstance(scores, list) and isinstance(scores[0], np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected List[np.ndarray] or np.ndarray, got {type(scores)}')
    cmaxs: List[np.ndarray] = []
    for score in scores:
        cmaxs.append(cummax(score))
    return cmaxs


def get_cummin(scores: List[np.ndarray]) -> List[np.ndarray]:
    """ Compute cumulative min for each array in a list

    Args:
        scores: list of the arrays on which `cummin` will be applied

    Returns:
        cmins:
    """
    if isinstance(scores, list) and isinstance(scores[0], np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected List[np.ndarray] or np.ndarray, got {type(scores)}')
    cmins: List[np.ndarray] = []
    for score in scores:
        cmins.append(-cummax(-score))
    return cmins


def plot_mean_std(*args, n_std: Optional[int] = 1,
                  ax: Optional[Axes] = None, alpha: float = .3, label: Optional[str] = None,
                  stop_plot_at_level: Optional[float] = None,
                  top_one_percent: Optional[bool] = False, minimisation: Optional[bool] = False,
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

    if not top_one_percent:
        mean = Y.mean(0)
        std = Y.std(0)
        if stop_plot_at_level is not None:
            idx = np.where(mean <= stop_plot_at_level)[0]
            std = std[idx]
            mean = mean[idx]
            # print(f"stop level reached at {len(idx)}")
        else:
            idx = X

        if ax is None:
            ax = plt.subplot()

        line_plot = ax.plot(X[idx], mean, label=label, **plot_mean_kwargs)

        if n_std > 0 and Y.shape[0] > 1:
            ax.fill_between(X[idx], mean - n_std * std, mean + n_std * std, alpha=alpha, color=line_plot[0].get_c())
    else:
        if minimisation:
            Y_top = Y.min(0)
        else:
            Y_top = Y.max(0)

        if ax is None:
            ax = plt.subplot()

        line_plot = ax.plot(X, Y_top, label=label, **plot_mean_kwargs)
    return ax


def plot_results(path_to_res, maximisation: bool, n_acqs: Optional[int] = None, ax: Optional[Axes] = None,
                 seeds: List[int] = None, **plt_kwargs):
    """

    Args:
        path_to_res: name to directory where results are saved in subdirectories corresponding to seeds
        maximisation: whether the task is a maximisation or minimisation
        n_acqs: number of acquisition steps over which to plot the results (if None all acquisitions are considered)
        ax: axis object
        seeds: list of seeds to consider (all if `None`)
        **plt_kwargs: plot options

    Returns:

    """
    results: List[np.array] = []
    if seeds is not None:
        paths: List[str] = [os.path.join(path_to_res, f"seed{i}") for i in seeds]
    else:
        paths: List[str] = glob.glob(path_to_res + '/seed*')

    for path_to_res in paths:
        path = os.path.join(path_to_res, 'results.npz')
        result = np.load(path, allow_pickle=True)
        results.append(result['opt_point_properties'][:n_acqs])

    if maximisation:
        regret = get_cummax(results)
    else:
        regret = get_cummin(results)
    return plot_mean_std(regret, ax=ax, **plt_kwargs)
