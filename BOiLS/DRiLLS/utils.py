# 2021.11.10-Add get_playground_dir and get_model_path
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
from inspect import signature
from typing import Callable, Any, Optional, Tuple
import numpy as np
from utils.utils_save import get_storage_root
import pandas as pd


def get_playground_dir(design: str, learner_id: str, root: str = get_storage_root(), seed: Optional[int] = None) -> str:
    playground_dir = os.path.join(root, 'playground', design, learner_id)
    if seed is not None:
        playground_dir = os.path.join(playground_dir, str(seed))
    return playground_dir


def get_model_path(design: str, learner_id: str, root: str = get_storage_root()) -> str:
    return os.path.join(root, 'model', learner_id, f"{design}", 'checkpoints')


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    r"""Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def get_info_log(log_file: str, iterations: int = None) -> Optional[Tuple[float, float, int, int]]:
    """
    Args:
        log_file: path to log file
        iterations: expected max number of iterations

    Returns:
        Infos relative to best design meeting constraint delay: area, delay, episode, iteration
    """
    logs = pd.read_csv(log_file)
    logs.columns = list(map(lambda col: col.replace(' ', ''), logs.columns))
    best_infos = logs["best_area_meets_constraint"].str.split("; ", expand=True)
    if iterations is not None and len(best_infos) != iterations:
        return None

    # making separate first name column from new data frame
    logs["best_area_meets_constraint-area"] = best_infos[0]

    # making separate last name column from new data frame
    logs["best_area_meets_constraint-delay"] = best_infos[1]
    logs["best_area_meets_constraint-best_episode"] = best_infos[2]
    logs["best_area_meets_constraint-best_iteration"] = best_infos[3]
    # Dropping old Name columns
    logs.drop(columns=["best_area_meets_constraint"], inplace=True)

    b_area = float(logs["best_area_meets_constraint-area"].iloc[-1])
    b_delay = float(logs["best_area_meets_constraint-delay"].iloc[-1])
    b_episode = int(logs["best_area_meets_constraint-best_episode"].iloc[-1])
    b_iteration = int(logs["best_area_meets_constraint-best_iteration"].iloc[-1])
    return b_area, b_delay, b_episode, b_iteration


def softmax(x: np.ndarray):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
