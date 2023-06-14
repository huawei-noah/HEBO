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

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.tasks.eda_seq_opt.eda_seq_opt_task import EDASeqOptimization
from mcbo.utils.general_utils import save_w_pickle

import torch

from mcbo.optimizers import GeneticAlgorithm

if __name__ == "__main__":
    from mcbo.task_factory import task_factory

    parser = argparse.ArgumentParser(add_help=True, description='Plot mixed optimisation results.')

    parser.add_argument("--design_name", "-d", required=True, help="Circuit name")

    args = parser.parse_args()

    pop_size = 25
    task_kwargs = {
        'designs_group_id': args.design_name,
        "operator_space_id": "basic",
        "objective": "both",
        "seq_operators_pattern_id": "basic",
        "n_parallel": pop_size
    }

    dtype = torch.float32
    aux = task_factory(task_name='aig_optimization', dtype=dtype, **task_kwargs)
    task: EDASeqOptimization = aux[0]
    search_space = aux[1]

    optimizer = GeneticAlgorithm(
        search_space=search_space, store_observations=True, input_constraints=None
    )
    print(f"{optimizer.name}_{task.name}")

    while task.num_func_evals < 500:
        x_next = optimizer.suggest(pop_size)
        y_next = task(x_next)
        optimizer.observe(x_next, y_next)
        print(f'Iteration {task.num_func_evals} - Best f(x) {optimizer.best_y:.3f}')

    save_path = os.path.join(Path(os.path.realpath(__file__)).parent.parent.parent.resolve(), "eda_ga25_boils_data",
                             task_kwargs["designs_group_id"], "data.pkl")
    task_ckpt_data = task.get_ckpt_data()
    data = {}
    data["objective"] = task_kwargs["objective"]
    data["ref_count"] = task.refs_1[0]
    data["level_ref"] = task.refs_2[0]
    data["X"] = task_ckpt_data["X"]
    data["Count"] = task_ckpt_data["intermediate_objs_1"][:, 0, -1]
    data["Level"] = task_ckpt_data["intermediate_objs_2"][:, 0, -1]
    data["operator_space_id"] = task.operator_space_id
    data["seq_operators_pattern_id"] = task.seq_operators_pattern_id
    data["cat_to_cmd"] = {i: task.operator_space.all_operators[i]().op_str() for i in
                          range(len(task.operator_space.all_operators))}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_w_pickle(data, save_path)
