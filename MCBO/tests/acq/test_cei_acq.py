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
from gpytorch.kernels import ScaleKernel, MaternKernel

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT


def test_cei(augmented: bool = False):
    dtype = torch.float64
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    task = task_factory('sphere', dtype=dtype, num_dims=10, variable_type='num')
    search_space = task.get_search_space(dtype=dtype)

    x_train_pd = search_space.sample(1000)
    x_train = search_space.transform(x_train_pd)
    y_train = torch.tensor(task(x_train_pd))
    y_constr_1 = torch.tensor(np.abs(x_train_pd.values.sum(1))).reshape(-1, 1)
    lambda_1 = 7

    x_test_pd = search_space.sample(200)
    x_test = search_space.transform(x_test_pd)
    # y_test = torch.tensor(task(x_test_pd))

    kernel = ScaleKernel(MaternKernel(ard_num_dims=search_space.num_dims))
    kernel_constr_1 = ScaleKernel(MaternKernel(ard_num_dims=search_space.num_dims))

    model = ExactGPModel(search_space, 1, kernel=kernel, device=device, dtype=dtype)
    model.fit(x_train, y_train)

    model_constr = ExactGPModel(search_space, 1, kernel=kernel_constr_1, device=device, dtype=dtype)
    model_constr.fit(x_train, y_constr_1)

    acq_func_id = "caei" if augmented else "cei"
    acq_func: CEI = acq_factory(acq_func_id=acq_func_id, num_constr=2)  # will consider a dummy constraint

    acq_res = acq_func(x_test, model, best_y=y_train.min(), constr_models=[model_constr, model_constr],
                       out_upper_constr_vals=torch.tensor([lambda_1, lambda_1]))

    print(acq_res)
    return 0


if __name__ == '__main__':
    from mcbo import task_factory
    from mcbo.acq_funcs.cei import CEI
    from mcbo.models.gp.exact_gp import ExactGPModel
    from mcbo.acq_funcs.factory import acq_factory

    for augmented_ in [False, True]:
        test_cei(augmented=augmented_)
