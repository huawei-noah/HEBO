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

import traceback

import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT
from core.utils.common_argparse import add_common_args

from core.algos.greedy.greedy_exp import ExpGreedy
from typing import Optional


def main(designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
         use_yosys: bool, objective: str,
         seed: int, overwrite: bool, n_parallel: int,
         lut_inputs: int,
         ref_abc_seq: Optional[str] = None):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        use_yosys: whether to use yosys command or abc python package for evaluation
        # abc_binary: (probably yosys-abc)
        n_parallel: number of threads to compute the refs
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
    """
    assert use_yosys, "No use_yosys not supported yet."
    exp: ExpGreedy = ExpGreedy(
        design_id=designs_group_id,
        lut_inputs=lut_inputs,
        max_iteration=seq_length,
        ref_abc_seq=ref_abc_seq,
        mapping=mapping,
        objective=objective,
        action_space_id=action_space_id,
        seed=seed,
        n_parallel=n_parallel,
    )
    if exp.already_trained_() and not args_.overwrite:
        exp.log(f"Experiment already trained: stored in {exp.playground_dir}")
        return
    elif exp.already_trained_():
        exp.log(f"Overwrite experiment: {exp.playground_dir}")
    result_dir = exp.playground_dir
    exp.log(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    logs = ''
    exc: Optional[Exception] = None
    try:
        exp.run()
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Performs logic synthesis optimization using Greedy search')
    # parser.register('type', list, parse_list)
    parser = add_common_args(parser)
    parser.add_argument("--n_parallel", type=int, default=1, help="number of threads to compute the stats")

    # Greedy Search
    parser.add_argument("--objective", type=str, choices=('lut', 'both', 'level', 'min_improvements'),
                        help="which objective should be optimized")
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")

    args_ = parser.parse_args()

    main(
        designs_group_id=args_.designs_group_id,
        seq_length=args_.seq_length,
        mapping=args_.mapping,
        action_space_id=args_.action_space_id,
        use_yosys=args_.use_yosys,
        seed=args_.seed,
        lut_inputs=args_.lut_inputs,
        ref_abc_seq=args_.ref_abc_seq,
        overwrite=args_.overwrite,
        objective=args_.objective,
        n_parallel=args_.n_parallel
    )
