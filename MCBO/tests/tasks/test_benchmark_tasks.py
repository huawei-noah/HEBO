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
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

if __name__ == "__main__":
    from mcbo.utils.experiment_utils import get_task_from_id
    from mcbo.utils.utils_task_test import test_task

    parser = argparse.ArgumentParser(add_help=True, description='Plot mixed optimization results.')

    parser.add_argument("--task_id", "-t", default=["pest"], type=str, nargs="+", help="Task ID")

    args = parser.parse_args()
    dtype = torch.float64
    for task_id in args.task_id:
        task = get_task_from_id(task_id=task_id)
        print(task_id)
        test_task(task=task, batch_size=1, n_evals=10)

