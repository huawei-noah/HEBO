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
from typing import Optional

import numpy as np

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import torch
from mcbo.optimizers.manual import Casmopolitan, BOiLS, COMBO, BODi, BOCS, BOSS, CoCaBO

from mcbo.optimizers.bo_builder import BO_ALGOS


def test_bo_builder(opt_name: Optional[str] = None):
    from mcbo.task_factory import task_factory
    n_init = 20

    task = task_factory('levy', num_dims=[2, 3, 2], variable_type=['num', 'nominal', 'nominal'],
                        num_categories=[None, 12, 12])
    search_space = task.get_search_space()

    input_constraints = None

    if opt_name is not None:
        BO_ALGOS[opt_name].build_bo(search_space=search_space, n_init=n_init, input_constraints=input_constraints)
        return 0

    for k, v in BO_ALGOS.items():
        try:
            v.build_bo(search_space=search_space, n_init=n_init, input_constraints=input_constraints)
        except Exception as e:
            print(k, e.args)
    return 0


def test_bo_builder_run(opt_name: str, mix: bool):
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
        n_cats = 11
        dim = 20
        task_kwargs = {'num_dims': dim, 'variable_type': 'nominal', 'num_categories': n_cats,
                       "task_name_suffix": None}

    task = task_factory(task_name, **task_kwargs)
    search_space = task.get_search_space(dtype=dtype)

    bo_n_init = 20
    bo_device = torch.device(f'cuda:{device_id}')
    opt_kwargs = dict(search_space=search_space, dtype=dtype, input_constraints=task.input_constraints)
    custom_builder = BO_ALGOS[opt_name]

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
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {custom_opt.best_y:.3f}')

    return 0


def test_bo_builder_v_hard_coded(opt_name: str, mix: bool):
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
        n_cats = 11
        dim = 20
        task_kwargs = {'num_dims': dim, 'variable_type': 'nominal', 'num_categories': n_cats,
                       "task_name_suffix": None}

    task = task_factory(task_name, **task_kwargs)
    search_space = task.get_search_space(dtype=dtype)

    bo_n_init = 20
    bo_device = torch.device(f'cuda:{device_id}')
    opt_kwargs = dict(search_space=search_space, dtype=dtype, input_constraints=task.input_constraints)
    custom_builder = BO_ALGOS[opt_name]

    np.random.seed(0)
    torch.manual_seed(0)
    custom_opt = custom_builder.build_bo(
        search_space=search_space,
        n_init=bo_n_init,
        input_constraints=task.input_constraints,
        dtype=dtype,
        device=bo_device
    )

    bo_opt_kwargs = dict(n_init=bo_n_init, device=bo_device, use_tr=custom_builder.tr_id is not None, **opt_kwargs)

    np.random.seed(0)
    torch.manual_seed(0)
    if opt_name == "Casmopolitan":
        hard_coded_opt = Casmopolitan(**bo_opt_kwargs)
    elif opt_name == "BOiLS":
        hard_coded_opt = BOiLS(**bo_opt_kwargs)
    elif opt_name == "COMBO":
        hard_coded_opt = COMBO(**bo_opt_kwargs)
    elif opt_name == "BODi":
        hard_coded_opt = BODi(**bo_opt_kwargs)
    elif opt_name == "BOCS":
        hard_coded_opt = BOCS(**bo_opt_kwargs)
    elif opt_name == "BOSS":
        hard_coded_opt = BOSS(**bo_opt_kwargs)
    elif opt_name == "CoCaBO":
        hard_coded_opt = CoCaBO(**bo_opt_kwargs)
    else:
        raise ValueError(opt_name)

    np.random.seed(0)
    torch.manual_seed(0)
    print(custom_opt.name)
    for i in range(n_evals):
        x_next = custom_opt.suggest(1)
        y_next = task(x_next)
        custom_opt.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {custom_opt.best_y:.3f}')

    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(n_evals):
        x_next = hard_coded_opt.suggest(1)
        y_next = task(x_next)
        hard_coded_opt.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {hard_coded_opt.best_y:.3f}')

    assert np.allclose(custom_opt.data_buffer.y.flatten().numpy(), hard_coded_opt.data_buffer.y.flatten().numpy()), (
        custom_opt.data_buffer.y.flatten().numpy(), hard_coded_opt.data_buffer.y.flatten().numpy())


def test_all_builder_v_hard_coded():
    mix = False
    for opt_name in list(BO_ALGOS.keys()):
        if opt_name == "CoCaBO":
            continue
        test_bo_builder_v_hard_coded(opt_name=opt_name, mix=mix)
    mix = True
    for opt_name in ["Casmopolitan", "BODi", "CoCaBO"]:
        test_bo_builder_v_hard_coded(opt_name=opt_name, mix=mix)


if __name__ == "__main__":
    device_id = 0
    test_bo_builder_run(opt_name="COMBO", mix=False)
    # test_all_builder_v_hard_coded()
