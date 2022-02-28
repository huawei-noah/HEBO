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
import numpy as np
import os
import sys
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT
from core.algos.genetic.sga.utils_sga import ResTuning

from utils.utils_misc import time_formatter, log
from core.algos.genetic.sga.multi_sga_exp import MultiSGARes
from core.utils.common_argparse import add_common_args
from core.algos.genetic.sga import main_multi_sga
from typing import Dict, Optional
import os
import time

from utils.utils_save import get_storage_tuning_root, load_w_pickle, save_w_pickle

MULTI_SGA_SPACE_1 = DesignSpace().parse(
    [
        {'name': 'parents_portion', 'type': 'num', 'lb': .05, 'ub': .6},
        {'name': 'elit_ration', 'type': 'num', 'lb': 0, 'ub': .05},
        {'name': 'mutation_probability', 'type': 'num', 'lb': 0, 'ub': .65},
        {'name': 'crossover_probability', 'type': 'num', 'lb': 0.2, 'ub': .8},
        {'name': 'crossover_type', 'type': 'cat',
         'categories': ['uniform', 'shuffle', 'segment', 'one_point', 'two_point']},
    ]
)

MULTI_SGA_SEARCH_SPACES: Dict[str, DesignSpace] = {
    'multi_sga_space_1': MULTI_SGA_SPACE_1
}


def get_multi_sga_tuning_path(search_space_id: str,
                              designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                              pop_size: int, seed: int, n_total_evals: int,
                              lut_inputs: int,
                              ref_abc_seq: str, objective: str):
    """
    Args:
        search_space_id: id of hebo search space
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        pop_size: population size for SGA
        n_total_evals: number of sequences to evaluate
        seed: reproducibility seed
        objective: quantity to optimize, either lut, level, or both
    """
    aux = f'search-{search_space_id}'
    aux += f"_{mapping}{f'-{lut_inputs}' if mapping == 'fpga' else ''}" \
           f"_seq-{seq_length}_ref-{ref_abc_seq}_act-{action_space_id}"
    aux += f'_pop-{pop_size}_n-tot-{n_total_evals}'
    return os.path.join(get_storage_tuning_root(), 'SGA', designs_group_id, aux, objective, str(seed))


def main(search_space_id: str, n_acq: int,
         designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
         library_file: str,
         abc_binary: str,
         pop_size: int, seed: int, n_total_evals: int, n_parallel: int,
         lut_inputs: int,
         ref_abc_seq: str, objective: str, overwrite: bool = False):
    """
    Args:
        search_space_id: id of hebo search space
        n_acq: number of configs to acquire with hebo
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
        n_total_evals: number of sequences to evaluate
        seed: reproducibility seed
        objective: which objective to optimize ('lut', 'level', 'both')
    """
    time_ref = time.time()
    search_space = MULTI_SGA_SEARCH_SPACES[search_space_id]
    res_path: str = os.path.join(
        get_multi_sga_tuning_path(
            search_space_id=search_space_id,
            designs_group_id=designs_group_id,
            seq_length=seq_length,
            mapping=mapping,
            action_space_id=action_space_id,
            pop_size=pop_size,
            seed=seed,
            n_total_evals=n_total_evals,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq,
            objective=objective
        ), 'res.pkl')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)

    def custom_log(msg_) -> None:
        log(msg_, header=f'Mutli-SGA tuning - {designs_group_id} - {objective}')

    custom_log(f"\n --- Start Tuning --- \nWill store results in: {res_path}")

    best_X = np.zeros((0, seq_length))
    best_F = np.zeros((0, 2))
    hebo_opt = HEBO(space=search_space)
    if os.path.exists(res_path) and not overwrite:
        res: ResTuning = load_w_pickle(os.path.dirname(res_path), os.path.basename(res_path))
        hebo_opt.observe(res.X_conf, res.y_score)
        best_X = res.best_X
        best_F = res.best_F
        custom_log(f"Loaded {hebo_opt.y.shape[0]} entries from {res_path}")

    logs = ''
    exc: Optional[Exception] = None
    try:
        while hebo_opt.y.shape[0] < n_acq:
            custom_log(
                f"Test {hebo_opt.y.shape[0] + 1} / {n_acq} --- {time_formatter(time.time() - time_ref)} since start")
            new = hebo_opt.suggest(1)
            assert len(new) == 1, len(new)
            exp_res_path: str = main_multi_sga.main(
                designs_group_id=designs_group_id,
                seq_length=seq_length,
                mapping=mapping,
                action_space_id=action_space_id,
                library_file=library_file,
                abc_binary=abc_binary,
                seed=seed + hebo_opt.y.shape[0],
                lut_inputs=lut_inputs,
                ref_abc_seq=ref_abc_seq,
                pop_size=pop_size,
                n_total_evals=n_total_evals,
                parents_portion=new['parents_portion'].iloc[0],
                mutation_probability=new['mutation_probability'].iloc[0],
                elit_ration=new['elit_ration'].iloc[0],
                crossover_probability=new['crossover_probability'].iloc[0],
                crossover_type=new['crossover_type'].iloc[0],
                overwrite=overwrite,
                n_parallel=n_parallel,
                objective=objective,
                verbose=False
            )
            exp_res: MultiSGARes = load_w_pickle(exp_res_path, 'res.pkl')
            if objective == 'both':
                best_ind = exp_res.F.sum(-1).argmin()
                best = exp_res.F[best_ind].sum()
            elif objective == 'lut':
                best_ind = exp_res.F[:, 0].argmin()
                best = exp_res.F[:, 0].min()
            elif objective == 'both':
                best_ind = exp_res.F[:, 1].argmin()
                best = exp_res.F[:, 1].min()
            else:
                raise ValueError(objective)
            best_X = np.vstack([best_X, exp_res.X[best_ind]])
            best_F = np.vstack([best_F, exp_res.F[best_ind]])

            score = best  # to be minimized

            hebo_opt.observe(new, np.array([score]))

            save_w_pickle(ResTuning(X_conf=hebo_opt.X, y_score=hebo_opt.y, best_X=best_X, best_F=best_F),
                          os.path.dirname(res_path), os.path.basename(res_path))
        custom_log(f"Saved results: {res_path}\n"
                   f"Finished: best {objective} improvement: {(hebo_opt.y.min()) * 100:.2f}")

    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(os.path.dirname(res_path), 'logs.txt'), "a")
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Performs tuning of Multi-SGA with HEBO')
    parser = add_common_args(parser)

    parser.add_argument("--n_parallel", type=int, default=1, help="number of threads to compute the stats")

    # SGA Search
    parser.add_argument("--pop_size", type=int, default=50, help="population size for SGA")
    parser.add_argument("--n_total_evals", type=int, required=True, help="number of sequences to evaluate")

    parser.add_argument("--n_acq", type=int, required=True, help="number of acquisitions with HEBO")
    parser.add_argument("--search_space_id", type=str, default='multi_sga_space_1', help="Id of the search space")

    parser.add_argument("--objective", type=str, choices=('lut', 'both', 'level'), help="which objective should "
                                                                                        "be optimized")
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.library_file):
        args_.library_file = os.path.join(ROOT_PROJECT, args_.library_file)

    main(
        search_space_id=args_.search_space_id,
        n_acq=args_.n_acq,
        designs_group_id=args_.designs_group_id,
        seq_length=args_.seq_length,
        mapping=args_.mapping,
        action_space_id=args_.action_space_id,
        library_file=args_.library_file,
        abc_binary=args_.abc_binary,
        seed=args_.seed,
        lut_inputs=args_.lut_inputs,
        ref_abc_seq=args_.ref_abc_seq,
        pop_size=args_.pop_size,
        n_total_evals=args_.n_total_evals,
        n_parallel=args_.n_parallel,
        overwrite=args_.overwrite,
        objective=args_.objective
    )
