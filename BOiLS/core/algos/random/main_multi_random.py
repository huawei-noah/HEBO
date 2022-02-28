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

from core.algos.random.multi_random_exp import MultiRandomExp
from typing import Optional


def main(designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
         library_file: str,
         abc_binary: str, use_yosys: bool,
         random_sampling_id: str, n_trials: int,
         seed: int, overwrite: bool, n_parallel: int,
         lut_inputs: int,
         ref_abc_seq: Optional[str] = None):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        library_file: library file (asap7.lib)
        use_yosys: whether to use yosys command or abc python package for evaluation
        abc_binary: (probably yosys-abc)
        n_parallel: number of threads to compute the refs
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        n_trials: number of random sequence to test
        random_sampling_id: id of the sampling technique used
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
    """
    exp: MultiRandomExp = MultiRandomExp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        mapping=mapping,
        action_space_id=action_space_id,
        library_file=library_file,
        use_yosys=use_yosys,
        abc_binary=abc_binary,
        n_trials=n_trials,
        seed=seed,
        n_parallel=n_parallel,
        random_sampling_id=random_sampling_id,
        lut_inputs=lut_inputs,
        ref_abc_seq=ref_abc_seq
    )
    exist = exp.exists()
    if exist and not overwrite:
        exp.log(f"Experiment already trained: stored in {exp.exp_path()}")
        return
    elif exist:
        exp.log(f"Overwrite experiment: {exp.exp_path()}")
    result_dir = exp.exp_path()
    exp.log(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    logs = ''
    exc: Optional[Exception] = None
    try:
        res = exp.run(n_parallel=n_parallel, overwrite=overwrite)
        exp.save_results(res)
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
                                     description='Performs logic synthesis optimization using NSAG2')
    # parser.register('type', list, parse_list)
    parser = add_common_args(parser)
    parser.add_argument("--n_parallel", type=int, default=1, help="number of threads to compute the stats")

    # Random Search
    parser.add_argument("--random_sampling_id", type=str, default='latin-hypercube',
                        choices=('latin-hypercube', 'random'), help="Type of random search")
    parser.add_argument("--n_trials", type=int, required=True, help="number of random sequence to test")
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.library_file):
        args_.library_file = os.path.join(ROOT_PROJECT, args_.library_file)

    main(
        designs_group_id=args_.designs_group_id,
        seq_length=args_.seq_length,
        mapping=args_.mapping,
        action_space_id=args_.action_space_id,
        library_file=args_.library_file,
        use_yosys=args_.use_yosys,
        abc_binary=args_.abc_binary,
        seed=args_.seed,
        lut_inputs=args_.lut_inputs,
        ref_abc_seq=args_.ref_abc_seq,
        random_sampling_id=args_.random_sampling_id,
        n_trials=args_.n_trials,
        overwrite=args_.overwrite,
        n_parallel=args_.n_parallel
    )
