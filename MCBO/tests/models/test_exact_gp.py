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

import torch

from mcbo.models.gp.exact_gp import ExactGPModel
from mcbo.models.gp.kernels import TransformedOverlap
from mcbo.task_factory import task_factory
from mcbo.utils.plotting_utils import plot_model_prediction

if __name__ == '__main__':
    dtype = torch.float64
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    task = task_factory('sphere', num_dims=10, variable_type='nominal', num_categories=10)
    search_space = task.get_search_space(dtype=dtype)

    x_train_pd = search_space.sample(1000)
    x_train = search_space.transform(x_train_pd)
    y_train = torch.tensor(task(x_train_pd))

    x_test_pd = search_space.sample(200)
    x_test = search_space.transform(x_test_pd)
    y_test = torch.tensor(task(x_test_pd))

    kernel = TransformedOverlap()

    model = ExactGPModel(search_space, 1, kernel, dtype=dtype, device=device)

    model.fit(x_train, y_train)

    plot_model_prediction(model, x_test, y_test, './model_pred.png')
