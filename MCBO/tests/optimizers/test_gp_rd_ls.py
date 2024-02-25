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

import numpy as np
import torch

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT


from mcbo.optimizers.bo_builder import BoBuilder


def test_gprd_builder_run(mix: bool):
    from mcbo.task_factory import task_factory
    n_evals = 200
    dtype = torch.float64

    task_name = "ackley"
    if mix:
        num_dims = [10, 5]
        variable_type = ['nominal', 'num']
        num_categories = [3, None]
        task_name_suffix = " 10-nom-3 5-num"
        lb = np.zeros(15)
        lb[-5:] = -1
        task_kwargs = dict(num_dims=num_dims, variable_type=variable_type, num_categories=num_categories,
                           task_name_suffix=task_name_suffix, lb=lb, ub=1)
    else:
        n_cats = 7
        dim = 10
        task_kwargs = {'num_dims': dim, 'variable_type': 'nominal', 'num_categories': n_cats,
                       "task_name_suffix": None}

    task = task_factory(task_name, **task_kwargs)
    search_space = task.get_search_space(dtype=dtype)

    bo_n_init = 20
    bo_device = torch.device(f'cuda:{device_id}')
    custom_builder = BoBuilder(model_id="gp_rd", acq_opt_id="mp", acq_func_id="addlcb", tr_id="basic")
    # custom_builder = BoBuilder(model_id="gp_rd", acq_opt_id="mab", acq_func_id="ei", tr_id="basic")

    np.random.seed(0)
    torch.manual_seed(0)
    custom_opt = custom_builder.build_bo(
        search_space=search_space,
        n_init=bo_n_init,
        input_constraints=task.input_constraints,
        dtype=dtype,
        device=bo_device
    )

    np.random.seed(0)
    torch.manual_seed(0)
    print(custom_opt.name)
    for i in range(n_evals):
        x_next = custom_opt.suggest(1)
        y_next = task(x_next)
        custom_opt.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {custom_opt.best_y[0]:.3f}')

    return 0


if __name__ == "__main__":
    device_id = 2
    test_gprd_builder_run(mix=False)
