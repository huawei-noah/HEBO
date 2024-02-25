# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import pathlib
import pickle
import random
import time
import warnings
from datetime import datetime
from inspect import signature
from typing import Any, Callable, Dict
from typing import Optional, List, Union, Tuple
from typing import TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from scipy.stats import t
from tqdm import tqdm

T = TypeVar('T')

matplotlib.use('Agg')


def get_project_root() -> str:
    return str(pathlib.Path(__file__).parent.parent.parent.resolve())


def load_yaml(path_to_yaml_file) -> Any:
    import yaml
    with open(path_to_yaml_file, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(dictionary, save_path) -> None:
    import yaml
    with open(save_path, 'w') as file:
        _ = yaml.dump(dictionary, file)


def set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_save_dir(save_dir) -> None:
    if os.path.exists(save_dir):
        print(
            f'{current_time_formatter()} - Directory {save_dir} already exists. '
            f'Continuing with the experiment may lead to previous results being overwritten.'
        )
    os.makedirs(save_dir, exist_ok=True)


def get_path_to_save_dir(settings) -> str:
    path = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "results", settings.get("task_name"),
                        settings.get("problem_name"), settings.get("save_dir"))
    return path


def array_to_tensor(x, device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64, device=device)
    if x.dim() == 1:
        x = x.reshape(1, -1)

    return x


def copy_tensor(x) -> torch.Tensor:
    return torch.empty_like(x).copy_(x)


def filter_kwargs(function: Callable, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
    r"""Given a function, select only the arguments that are applicable.

    Returns:
         The kwargs dict containing only the applicable kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def save_w_pickle(obj: Any, path: str, filename: Optional[str] = None) -> None:
    """ Save object obj in file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_w_pickle(path: str, filename: Optional[str] = None) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    p = os.path.join(path, filename)
    with open(p, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            raise Exception(f"EOFError with {p}")
        except UnicodeDecodeError:
            raise Exception(f"UnicodeDecodeError with {p}")
        except pickle.UnpicklingError:
            raise Exception(f"UnpicklingError with {p}")


def safe_load_w_pickle(path: str, filename: Optional[str] = None, n_trials=3, time_sleep=2) -> Any:
    """ Make several attempts to load object from file exp_path/filename.pkl """
    trial = 0
    end = False
    result = None
    while not end:
        try:
            result = load_w_pickle(path=path, filename=filename)
            end = True
        except (pickle.UnpicklingError, EOFError) as e:
            trial += 1
            if trial > n_trials:
                raise e
            time.sleep(time_sleep)
        except UnicodeDecodeError as e:
            if filename is None:
                filename = os.path.basename(path)
                path = os.path.dirname(path)
            log(os.path.join(path, filename))
            raise e
    return result


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


def current_time_formatter() -> str:
    return "{:%Y-%m-%d %H:%M:%S}".format(datetime.now())


def log(message, header: Optional[str] = None, end: Optional[str] = None) -> None:
    if header is None:
        header = ''
    print(f'[{header}' + f' {current_time_formatter()}' + f"]  {message}", end=end)


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


def get_common_chunk_sizes(ys: List[np.ndarray]) -> List[Tuple[np.ndarray, List[List[int]]]]:
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
        x_s = np.arange(lens[i - 1], lens[i])
        y = [y[lens[i - 1]:lens[i]] for y in ys if len(y) >= lens[i]]
        output.append((x_s, y))
    return output


def plot_mean_std(*args, n_std: Optional[float] = 1,
                  ax: Optional[Axes] = None, alpha: float = .3, errbar: bool = False,
                  lb: Optional[Union[float, np.ndarray]] = None,
                  ub: Optional[Union[float, np.ndarray]] = None,
                  linewidth: int = 3,
                  show_std_error: Optional[bool] = False,
                  ci_level: Optional[float] = None,
                  **plot_mean_kwargs) -> Axes:
    """ Plot mean and std (with fill between) of sequential data Y of shape (n_trials, lenght_of_a_trial)

    Args:
        args: x-values (if None, we will take `range(0, len(Y))`), y-values
        n_std: number of std to plot around the mean (if `0` only the mean is plotted)
        ax: axis on which to plot the curves
        alpha: parameter for `fill_between`
        errbar: use error bars instead of shaded area
        ci_level: show confidence interval over the mean at specified level (e.g. 0.95), otherwise uncertainty shows
          n_std std around the mean
        linewidth: width of line
        lb: lower bound (to clamp uncertainty region)
        ub: upper bound (to clamp uncertainty region)
        show_std_error: show standard error (std / sqrt(n_samples)) as shadow area around mean curve

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

    if len(x) == 0:
        return ax

    if ci_level is not None and len(y) > 1:
        # student
        t_crit = np.abs(t.ppf((1 - ci_level) / 2, len(y) - 1))
        n_std = t_crit / np.sqrt(len(y))
    elif show_std_error:
        n_std = 1 / np.sqrt(len(y))

    if errbar:
        n_errbars = min(10, len(std))
        errbar_inds = len(std) // n_errbars
        ax.errorbar(x, mean, yerr=n_std * std, errorevery=errbar_inds, linewidth=linewidth, **plot_mean_kwargs)
    else:
        line_plot = ax.plot(x, mean, linewidth=linewidth, **plot_mean_kwargs)

        if n_std > 0 and y.shape[0] > 1:
            uncertainty_lb = mean - n_std * std
            uncertainty_ub = mean + n_std * std
            if lb is not None:
                uncertainty_lb = np.maximum(uncertainty_lb, lb)
            if ub is not None:
                uncertainty_ub = np.minimum(uncertainty_ub, ub)

            ax.fill_between(x, uncertainty_lb, uncertainty_ub, alpha=alpha, color=line_plot[0].get_c())

    return ax


def filter_nans(x: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Eliminate NaNs

    Args:
        x: points in search space
        y: 2-d array of black-box values

    Returns:
        x: entries whose associated y is not NaN
        y: black-box values that are not NaN
    """
    num_nan = np.isnan(y).sum()
    if num_nan > 0:
        warnings.warn(
            f"Got {num_nan} / {len(y)} NaN observations.\n"
            f"X:\n"
            f"    {x}\n"
            f"Y:\n"
            f"    {y}"
        )

    filtr_ind = np.arange(len(y))[np.isnan(y).sum(-1) == 0]
    y = y[filtr_ind]
    x = x.iloc[filtr_ind]

    return x, y


def parallel_call_wrapper(func: Callable[[Any], T], func_kwargs_list: List[Dict[str, Any]], n_parallel=None) -> List[T]:
    if n_parallel is None:
        n_parallel = len(os.sched_getaffinity(0)) - 1
    if n_parallel == 1:  # do not run in parallel -> this will make debugging easier
        return [func(**func_kwargs) for func_kwargs in tqdm(func_kwargs_list)]
    return Parallel(n_jobs=n_parallel, backend="multiprocessing")(
        delayed(func)(**func_kwargs) for func_kwargs in
        tqdm(func_kwargs_list)
    )


def geo_mean_overflow(iterable) -> float:
    """ Geometric mean """
    return np.exp(np.log(iterable).mean())
