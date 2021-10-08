from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Union, Tuple, List


def get_project_root() -> str:
    """ return path of project root """
    return str(Path(__file__).parent.parent)


def get_timestr(fmt: str = "%Y%m%d-%H%M%S") -> str:
    now = datetime.now()  # current date and time
    return str(now.strftime(fmt))

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