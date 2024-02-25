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
from typing import Dict, Callable

import numpy as np
import torch

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT


def input_constraint_maker(ind: int) -> Callable[[Dict], bool]:
    def f(x: Dict) -> bool:
        return x[f"var_{ind}"] < 0

    return f


def test_rs_acq_opt_in_bobuilder() -> int:
    acq_opt_id = "rs"
    bo_builder = BoBuilder(model_id="gp_to", acq_opt_id=acq_opt_id, acq_func_id="ei", tr_id="basic")
    optimizer = bo_builder.build_bo(
        search_space=search_space, n_init=10, input_constraints=input_constraints, dtype=dtype, device=device
    )
    print(optimizer.name, task.name)
    for i in range(20):
        x_next = optimizer.suggest(3)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {i + 1:>4d} - f(x) - {y_next[0][0]:.3f} - f(x*) - {optimizer.best_y:.3f}')

    return 0


def test_rs_acq_opt() -> int:
    x_train_pd = search_space.sample(100)
    x_train = search_space.transform(x_train_pd)
    y_train = torch.tensor(task(x_train_pd))

    kernel = TransformedOverlap()
    model = ExactGPModel(search_space, 1, kernel, dtype=dtype, device=device)
    model.fit(x_train, y_train)

    acq_opt = RandomSearchAcqOptimizer(
        search_space=search_space,
        input_constraints=input_constraints,
        obj_dims=[0],
        out_upper_constr_vals=None,
        out_constr_dims=None
    )

    x = search_space.transform(search_space.sample(1))
    acq_func = acq_factory("lcb")
    new_points = acq_opt.optimize(
        x=x,
        n_suggestions=3,
        x_observed=x_train,
        model=model,
        acq_func=acq_func,
        acq_evaluate_kwargs={},
        tr_manager=None
    )
    new_points = search_space.inverse_transform(new_points)
    assert np.all(
        [[input_constraints[i](new_points.iloc[[j]]) for i in range(len(input_constraints))] for j in
         range(len(new_points))])
    print(new_points)
    return 0


if __name__ == '__main__':
    from mcbo.task_factory import task_factory
    from mcbo.optimizers import BoBuilder
    from mcbo.acq_funcs import acq_factory
    from mcbo.acq_optimizers.random_search_acq_optimizer import RandomSearchAcqOptimizer
    from mcbo.models import ExactGPModel
    from mcbo.models.gp.kernels import TransformedOverlap

    dtype = torch.float64
    device = torch.device("cuda")
    task = task_factory('levy', num_dims=[2, 3, 2], variable_type=['int', 'nominal', 'num'],
                        num_categories=[None, 21, None])
    search_space = task.get_search_space(dtype=dtype)

    input_constraints = [input_constraint_maker(i) for i in range(1, 4)]

    test_rs_acq_opt()
