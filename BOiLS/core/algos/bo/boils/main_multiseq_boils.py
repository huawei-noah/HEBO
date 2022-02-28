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

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from utils.utils_save import load_w_pickle
from core.utils.common_argparse import add_common_args
from core.algos.bo.boils.multiseq_boils_exp import MultiseqBoilsExp
from typing import Optional


def main(designs_group_id: str, seq_length: int, n_universal_seqs: int, mapping: str, action_space_id: str,
         library_file: str,
         abc_binary: str,
         seed: int, n_initial: int, standardise: bool, ard: bool, acq: str,
         n_total_evals: int,
         overwrite: bool, n_parallel: int,
         lut_inputs: int,
         ref_abc_seq: str, objective: str, device: Optional[int], verbose: bool = True):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        n_universal_seqs: number of sequences
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        n_parallel: number of threads to compute the refs
        library_file: library file (asap7.lib)
        abc_binary: (probably yosys-abc)
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        n_total_evals: number of sequences to evaluate
        n_initial: number of initial points to test before building first surrogate model
        standardise: whether to standardise the ys for the GP fit
        ard: whether to enable automatic relevance determination
        acq: choice of the acquisition function.
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
        objective: which objective to optimize ('lut', 'level', 'both')
        device: gpu device id
    """
    exp: MultiseqBoilsExp = MultiseqBoilsExp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        n_universal_seqs=n_universal_seqs,
        mapping=mapping,
        action_space_id=action_space_id,
        library_file=library_file,
        abc_binary=abc_binary,
        seed=seed,
        n_initial=n_initial,
        standardise=standardise,
        ard=ard,
        acq=acq,
        lut_inputs=lut_inputs,
        ref_abc_seq=ref_abc_seq,
        n_parallel=n_parallel,
        objective=objective,
        overwrite=overwrite
    )
    exist = exp.exists()
    if exist and not overwrite:
        if not os.path.exists(os.path.join(exp.exp_path(), 'ckpt.pkl')):
            exist = False
        else:
            # check if enough sequences have been evaluated
            ckpt = load_w_pickle(exp.exp_path(), 'ckpt.pkl')
            if len(ckpt.full_objs_1) < n_total_evals:
                exist = False
    if exist and not overwrite:
        exp.log(f"Experiment already trained: stored in {exp.exp_path()}")
        return exp.exp_path()
    elif exist:
        exp.log(f"Overwrite experiment: {exp.exp_path()}")
    result_dir = exp.exp_path()
    exp.log(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    logs = ''
    exc: Optional[Exception] = None
    try:
        res = exp.run(n_total_evals=n_total_evals, verbose=verbose, device=device)
        exp.save_results(res)
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc
    return exp.exp_path()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Performs logic synthesis optimization using NSAG2')
    parser = add_common_args(parser)
    parser.add_argument("--n_parallel", type=int, default=1, help="number of threads to compute the stats")
    parser.add_argument("--n_universal_seqs", type=int, help="number of universal sequences to optimise")

    # Boilspolitan Search
    parser.add_argument("--n_total_evals", type=int, required=True, help="number of sequences to evaluate")
    parser.add_argument("--n_initial", type=int, default=20, help="Number of initial random points to evaluate")
    parser.add_argument("--standardise", action='store_true', help="whether to standardise the ys for the GP fit")
    parser.add_argument("--ard", action='store_true', help="whether to use ard")
    parser.add_argument("--acq", type=str,
                        choices=('ei', 'ucb'),
                        default='ei',
                        help="acquistion function")
    parser.add_argument("--objective", type=str, choices=('lut', 'both', 'level', 'min_improvements'),
                        help="which objective should be optimized")
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")
    parser.add_argument("--device", type=int, help="cuda id (cpu if None or negative)")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.library_file):
        args_.library_file = os.path.join(ROOT_PROJECT, args_.library_file)

    main(
        designs_group_id=args_.designs_group_id,
        seq_length=args_.seq_length,
        n_universal_seqs=args_.n_universal_seqs,
        mapping=args_.mapping,
        action_space_id=args_.action_space_id,
        library_file=args_.library_file,
        abc_binary=args_.abc_binary,
        seed=args_.seed,
        lut_inputs=args_.lut_inputs,
        ref_abc_seq=args_.ref_abc_seq,
        n_total_evals=args_.n_total_evals,
        standardise=args_.standardise,
        acq=args_.acq,
        ard=args_.ard,
        device=args_.device,
        n_initial=args_.n_initial,
        overwrite=args_.overwrite,
        n_parallel=args_.n_parallel,
        objective=args_.objective
    )
