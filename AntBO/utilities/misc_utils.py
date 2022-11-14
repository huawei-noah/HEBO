import os
import pickle
import time
from datetime import datetime
from inspect import signature
from typing import Optional, Callable, Any, Dict


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


def _filter_kwargs(function: Callable, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
        except EOFError as e:
            print(path, filename)
            raise
