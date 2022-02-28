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
import traceback
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from core.utils.common_argparse import add_common_args
from core.algos.GRiLLS.multi_grills_exp import MultiGRiLLSExp, MultiGRiLLSRes
from typing import Optional


def main(designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
         library_file: str,
         abc_binary: str,
         seed: int, alpha_pi: float, alpha_v: float, gamma: float, n_episodes: int,
         overwrite: bool,
         lut_inputs: int,
         objective: str,
         use_yosys: bool,
         ref_abc_seq: Optional[str] = None):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        library_file: library file (asap7.lib)
        abc_binary: (probably yosys-abc)
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        n_episodes: number of RL episodes to run
        seed: seed for reproducibility
        objective: which objective to optimize ('lut', 'level', 'both')
        alpha_pi: policy learning rate
        alpha_v: value function learning rate
        gamma: decaying rate
        overwrite: Overwrite existing experiment
    """
    exp: MultiGRiLLSExp = MultiGRiLLSExp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        mapping=mapping,
        action_space_id=action_space_id,
        library_file=library_file,
        abc_binary=abc_binary,
        objective=objective,
        use_yosys=use_yosys,
        seed=seed,
        lut_inputs=lut_inputs,
        ref_abc_seq=ref_abc_seq,
        alpha_pi=alpha_pi,
        alpha_v=alpha_v,
        n_episodes=n_episodes,
        gamma=gamma
    )
    exist = exp.exists()
    if exist and not overwrite:
        print(f"Experiment already trained: stored in {exp.exp_path()}")
        return
    elif exist:
        print(f"Overwrite experiment: {exp.exp_path()}")
    result_dir = exp.exp_path()
    print(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    logs = ''
    exc: Optional[Exception] = None
    try:
        res: MultiGRiLLSRes = exp.run()
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
    parser = add_common_args(parser)

    # GRiLLS params
    parser.add_argument("--alpha_pi", type=float, default=8e-4, help="policy learning rate")
    parser.add_argument("--alpha_v", type=float, default=3e-3, help="value function learning rate")
    parser.add_argument("--gamma", type=float, default=.9, help="decaying rate")
    parser.add_argument("--n_episodes", type=int, required=True, help="number of RL episodes to run")

    parser.add_argument("--objective", type=str, choices=('lut', 'both', 'level', 'min_improvements'),
                        help="which objective should be optimized")
    parser.add_argument("--seed", type=int, nargs='+', help="seed for reproducibility")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.library_file):
        args_.library_file = os.path.join(ROOT_PROJECT, args_.library_file)

    for seed_ in args_.seed:
        main(
            designs_group_id=args_.designs_group_id,
            seq_length=args_.seq_length,
            mapping=args_.mapping,
            action_space_id=args_.action_space_id,
            library_file=args_.library_file,
            abc_binary=args_.abc_binary,
            objective=args_.objective,
            use_yosys=args_.use_yosys,
            seed=seed_,
            lut_inputs=args_.lut_inputs,
            ref_abc_seq=args_.ref_abc_seq,
            alpha_pi=args_.alpha_pi,
            alpha_v=args_.alpha_v,
            gamma=args_.gamma,
            n_episodes=args_.n_episodes,
            overwrite=args_.overwrite
        )
