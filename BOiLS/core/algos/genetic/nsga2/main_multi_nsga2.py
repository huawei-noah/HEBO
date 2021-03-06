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

from core.utils.common_argparse import add_common_args
from core.algos.genetic.nsga2.multi_nsga2_exp import MultiNSGA2Exp
from typing import Optional


def main(designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
         library_file: str,
         abc_binary: str, use_yosys: bool,
         pop_size: int, seed: int, n_gen: int, eta_mutation: float,
         eta_cross: int, prob_cross: float, selection: str,
         overwrite: bool, n_parallel: int,
         lut_inputs: int,
         ref_abc_seq: str, verbose: bool = True):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        n_parallel: number of threads to compute the refs
        library_file: library file (asap7.lib)
        abc_binary: (probably yosys-abc)
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        pop_size: population size for SGA
        n_gen: number of generations
        eta_mutation: eta parameter for int_pm mutation
        eta_cross: eta parameter for crossover
        prob_cross: prob parameter for crossover
        selection: selection process
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
    """
    exp: MultiNSGA2Exp = MultiNSGA2Exp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        mapping=mapping,
        action_space_id=action_space_id,
        library_file=library_file,
        abc_binary=abc_binary,
        n_gen=n_gen,
        eta_mutation=eta_mutation,
        eta_cross=eta_cross,
        prob_cross=prob_cross,
        selection=selection,
        seed=seed,
        pop_size=pop_size,
        lut_inputs=lut_inputs,
        ref_abc_seq=ref_abc_seq,
        n_parallel=n_parallel,
        use_yosys=use_yosys
    )
    exist = exp.exists()
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
        res = exp.run(verbose=verbose)
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

    # SGA Search
    parser.add_argument("--pop_size", type=int, default=100, help="population size for SGA")
    parser.add_argument("--n_gen", type=int, help="Number of generations")
    parser.add_argument("--eta_cross", type=float, default=15, help="eta parameter for int_sbx crossover")
    parser.add_argument("--eta_mute", type=float, default=20, help="eta parameter for int_pm mutation")
    parser.add_argument("--prob_cross", type=float, default=0.9, help="Probability of crossover")
    parser.add_argument("--selection", type=str, default="random", choices=('random', 'tournament'),
                        help="Selection process.")
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
        abc_binary=args_.abc_binary,
        use_yosys=args_.use_yosys,
        seed=args_.seed,
        lut_inputs=args_.lut_inputs,
        ref_abc_seq=args_.ref_abc_seq,
        pop_size=args_.pop_size,
        n_gen=args_.n_gen,
        eta_cross=args_.eta_cross,
        eta_mutation=args_.eta_mute,
        prob_cross=args_.prob_cross,
        selection=args_.selection,
        overwrite=args_.overwrite,
        n_parallel=args_.n_parallel,
    )
