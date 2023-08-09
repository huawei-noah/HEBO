# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
import time
import numpy as np
import torch


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }


def get_global_name(name):
    """
    In case one modifies the environment.
    """
    if 'cartpole' in name:
        glob_name = 'cartpole'
    elif 'point_mass' in name:
        glob_name = 'point_mass'
    elif 'hirl_point_fall' in name:
        glob_name = 'hirl_point_fall'
    else:
        glob_name = name
    return glob_name


def get_global_agent_name(agent_name):
    """
    For variations of the same agent (for example, Naive or not).
    """
    glob_name = agent_name
    return glob_name
