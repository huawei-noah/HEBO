# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary
# forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import os.path
import numpy as np

from pathlib import Path
import torch

from mcbo.optimizers.manual.bodi import BODi
from mcbo.utils.plotting_utils import plot_convergence_curve
from typing import Dict, Callable


def input_constraint_maker(ind: int) -> Callable[[Dict], bool]:
    def f(x: Dict) -> bool:
        return x[f"var_{ind}"] < 0

    return f


if __name__ == '__main__':
    from mcbo.task_factory import task_factory

    dtype=torch.float64
    task = task_factory('levy', num_dims=[2, 3, 2], variable_type=['int', 'nominal', 'num'],
                                      num_categories=[None, 21, None])
    search_space = task.get_search_space(dtype=dtype)


    task_name = "ackley"
    num_dims = [50, 3]
    variable_type = ['nominal', 'num']
    num_categories = [2, None]
    task_name_suffix = " 50-nom-2 3-num"
    lb = np.zeros(53)
    lb[-3:] = -1
    task_kwargs = dict(num_dims=num_dims, variable_type=variable_type, num_categories=num_categories,
                       task_name_suffix=task_name_suffix, lb=lb, ub=1)
    task = task_factory(task_name=task_name, **task_kwargs)
    search_space = task.get_search_space(dtype=dtype)

    input_constraints = None  # [input_constraint_maker(i) for i in range(1, 4)]

    optimizer = BODi(
        search_space=search_space,
        n_init=20,
        use_tr=False,
        acq_optim_n_iter=100,
        device=torch.device('cuda:0'),
        input_constraints=input_constraints
    )

    for i in range(200):
        x_next = optimizer.suggest(1)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {optimizer.best_y:.3f}')

    plot_convergence_curve(optimizer, task, os.path.join(Path(os.path.realpath(__file__)).parent.parent.resolve(),
                                                         f'{optimizer.name}_test.png'), plot_per_iter=True)
