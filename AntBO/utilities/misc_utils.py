import os
import pickle
import time
from datetime import datetime
from inspect import signature
from typing import Optional, Callable, Any, Dict, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def time_formatter(t: float, show_ms: bool = False) -> str:
    """ Convert a duration in seconds to a str `dd:hh:mm:ss`

    Args:
        t: time in seconds
        show_ms: whether to show ms on top of dd:hh:mm:ss
    """
    n_day = time.gmtime(t).tm_yday - 1
    if n_day > 0:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
        ts = f"{n_day}:{ts}"
    else:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
    if show_ms:
        ts += f'{t - int(t):.3f}'.replace('0.', '.')
    return ts


def log(message, header: Optional[str] = None, end: Optional[str] = None):
    if header is None:
        header = ''
    print(f'[{header}' + ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) + f"]  {message}", end=end)


def filter_kwargs(function: Callable, **kwargs: dict[str, Any]) -> Dict[str, Any]:
    r"""Given a function, select only the arguments that are applicable.

    Return:
         The kwargs dict containing only the applicable kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def save_w_pickle(obj: Any, path: str, filename: Optional[str] = None) -> None:
    """ Save object obj in file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_w_pickle(path: str, filename: Optional[str] = None) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            print(path, filename)
            raise


def cummax(x: np.ndarray, return_ind=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Return array containing at index `i` the value max(X)[:i] """
    cmaxind: List[int] = [0]
    cmax: List[float] = [x[0]]
    for i, xx in enumerate(x[1:]):
        i += 1
        if xx > cmax[-1]:
            cmax.append(xx)
            cmaxind.append(i)
        else:
            cmax.append(cmax[-1])
            cmaxind.append(cmaxind[-1])
    cmax_np = np.array(cmax)
    assert np.all(x[cmaxind] == cmax_np), (x, x[cmaxind], cmax_np)
    if return_ind:
        return cmax_np, np.array(cmaxind)
    return cmax_np


def get_cummax(scores: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """ Compute cumulative max for each array in a list

    Args:
        scores: list of the arrays on which `cummax` will be applied

    Returns:
        cmaxs:
    """
    if not isinstance(scores, list) and isinstance(scores, np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected List[np.ndarray] or np.ndarray, got {type(scores)}')

    cmaxs: List[np.ndarray] = []
    for score in scores:
        cmaxs.append(cummax(score))
    return cmaxs


def get_cummin(scores: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """ Compute cumulative min for each array in a list

    Args:
        scores: list of the arrays on which `cummin` will be applied

    Returns:
        cmins:
    """
    if not isinstance(scores, list) and isinstance(scores, np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected List[np.ndarray] or np.ndarray, got {type(scores)}')
    cmins: List[np.ndarray] = []
    for score in scores:
        cmins.append(-cummax(-score))
    return cmins


def get_common_chunk_sizes(ys: List[np.ndarray]):
    """ From a list of arrays of various sizes, get a list of `list of arrays of same size`

     Example:
         >>> ys = [[1, 3 ,4, 5],
                   [0, 7, 8 , 2, 9],
                   [-1]]
         >>> get_common_chunk_sizes(ys)
         ---> [
         --->   ([0], [[1], [0], [-1]]),               # gather all elements of index in [0]
         --->   ([1, 2, 3], [[3, 4, 5], [7, 8, 2]]),   # gather all elements of index in [1, 2, 3]
         --->   ([4], [[9]])                           # gather all elements of index in [4]
         ---> ]
     """
    ys = [y for y in ys if len(y) > 0]
    lens = [0] + sorted(set([len(y) for y in ys]))

    output = []
    for i in range(1, len(lens)):
        xs = np.arange(lens[i - 1], lens[i])
        y = [y[lens[i - 1]:lens[i]] for y in ys if len(y) >= lens[i]]
        output.append((xs, y))
    return output


def plot_mean_std(*args, n_std: Optional[int] = 1,
                  ax: Optional[Axes] = None, alpha: float = .3,
                  **plot_mean_kwargs):
    """ Plot mean and std (with fill between) of sequential data Y of shape (n_trials, lenght_of_a_trial)

    Args:
        args: 
            x: x-values (if None, we will take `range(0, len(Y))`)
            y: y-values
        n_std: number of std to plot around the mean (if `0` only the mean is plotted)
        ax: axis on which to plot the curves
        alpha: parameter for `fill_between`

    Returns:
        The axis.
    """
    if len(args) == 1:
        y = args[0]
        x = None
    elif len(args) == 2:
        x, y = args
    else:
        raise RuntimeError('Wrong number of arguments (should be [X], Y,...)')

    assert len(y) > 0, 'Y should be a non-empty array, nothing to plot'
    y = np.atleast_2d(y)
    if x is None:
        x = np.arange(y.shape[1])
    assert x.ndim == 1, f'X should be of rank 1, got {x.ndim}'
    mean = y.mean(0)
    std = y.std(0)
    if ax is None:
        ax = plt.subplot()

    line_plot = ax.plot(x, mean, **plot_mean_kwargs)

    if n_std > 0 and y.shape[0] > 1:
        ax.fill_between(x, mean - n_std * std, mean + n_std * std, alpha=alpha, color=line_plot[0].get_c())

    return ax
