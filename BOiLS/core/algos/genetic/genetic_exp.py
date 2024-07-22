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
import time
from typing import Optional, List, Any, Dict, Tuple

import numpy as np
from abc import ABC
import pandas as pd
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling, get_mutation, get_crossover
from pymoo.interface import sample
from pymoo.model.problem import Problem
from pymoo.model.result import Result
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from utils.utils_misc import log
from core.action_space import Action
from core.algos.common_exp import EDAExp
from core.sessions.utils import get_design_prop
from utils.utils_save import save_w_pickle


class GeneticExp(EDAExp, ABC):
    linestyle = '-'

    meta_method_id_ = 'GA'

    def __init__(self, design_file: str, seq_length: int, mapping: str, action_space_id: str, library_file: str,
                 abc_binary: str, lut_inputs: int = 4, ref_abc_seq: Optional[str] = None):
        """
        Args:
            design_file: path to the design
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """
        super().__init__(design_file=design_file,
                         seq_length=seq_length,
                         mapping=mapping,
                         action_space_id=action_space_id,
                         library_file=library_file,
                         abc_binary=abc_binary,
                         lut_inputs=lut_inputs,
                         ref_abc_seq=ref_abc_seq)

    @property
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        return self.meta_method_id_


class NSGA2Exp(GeneticExp):
    color = 'red'

    def __init__(self, design_file: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 pop_size: int, seed: int, n_gen: int,
                 lut_inputs: int = 4,
                 ref_abc_seq: Optional[str] = None):
        """
        Args:
            design_file: path to the design
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            pop_size: population size for NSGA2
            n_gen: number of generations to run with
            seed: reproducibility seed
        """

        super().__init__(design_file=design_file, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, abc_binary=abc_binary,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq)
        self.pop_size = pop_size
        self.seed = seed
        self.n_gen = n_gen
        self.termination = get_termination("n_gen", self.n_gen)

        # pymoo problem
        self.problem = NSGA2Problem(
            n_var=seq_length,
            action_space=self.action_space,
            design_file=self.design_file,
            mapping=self.mapping,
            library_file=library_file,
            abc_binary=abc_binary,
            lut_inputs=self.lut_inputs,
            ref_1=self.ref_1,
            ref_2=self.ref_2
        )

        self.algo = NSGA2(pop_size=self.pop_size,
                          sampling=self.get_sampling(),
                          mutation=self.get_mutation(),
                          crossover=self.get_crossover(),
                          eliminate_duplicates=True)

    def get_config(self) -> Dict[str, Any]:
        config = super(NSGA2Exp, self).get_config()
        config['pop_size'] = self.pop_size
        config['n_gen'] = self.n_gen
        config['seed'] = self.seed

        return config

    def get_mutation(self):
        return get_mutation('int_pm', eta=20)

    def get_crossover(self):
        return get_crossover('int_sbx', eta=15, prob=0.9)

    def exp_id(self) -> str:
        return f'NSGA2_pop-{self.pop_size}_gen-{self.n_gen}'

    def exp_path(self) -> str:
        return os.path.join(super(NSGA2Exp, self).exp_path(), str(self.seed))

    def run(self, verbose: bool = False, visualize: bool = False):
        np.random.seed(self.seed)
        t = time.time()
        res = minimize(
            problem=self.problem,
            algorithm=self.algo,
            termination=self.termination,
            seed=self.seed,
            verobse=verbose,
            save_history=True
        )
        self.exec_time = time.time() - t
        if verbose:
            log(f"Took {self.exec_time} to optimise {self.design_name}")
        if visualize:
            plot = Scatter()
            plot.add(self.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.add(res.F, color="red")
            plot.show()

        return res

    def process_results(self, res) -> pd.DataFrame:
        seq_id = []
        obj_1 = []
        ratio_1 = []
        obj_2 = []
        ratio_2 = []
        for seq_ind, func_value in zip(res.X, res.F):
            seq_id.append(' | '.join([self.action_space[ind].act_id for ind in seq_ind]))
            ratio_1.append(func_value[0])
            ratio_2.append(func_value[1])
            obj_1.append(ratio_1[-1] * self.ref_1)
            obj_2.append(ratio_2[-1] * self.ref_2)
        pd_res = pd.DataFrame()
        pd_res['seq_id'] = seq_id

        pd_res[self.obj1_id] = obj_1
        pd_res[self.obj2_id] = obj_2

        pd_res['ratio ' + self.obj1_id] = ratio_1
        pd_res['ratio ' + self.obj2_id] = ratio_2

        pd_res['qor'] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values('qor')

    def get_sampling(self) -> np.ndarray:
        """ Return either an array of initial population obtained with latin hypercube sampling """
        sampling = get_sampling("int_lhs")
        samples = sample(sampling, n_samples=self.pop_size, n_var=self.seq_length, xl=0, xu=len(self.action_space) - 1)

        if self.seq_length == len(self.biseq.sequence):  # we look for a sequence of same length as the reference
            action_id_list = list(map(lambda action: action.act_id, self.action_space))
            ref_sequence = []
            for i, seq_id in enumerate(self.biseq.sequence):
                if seq_id not in action_id_list:
                    print(
                        f"{seq_id} not in available actions: \n{action_id_list}\n\t"
                        f"-> don't include this sequence in initialization")
                    break
                ind = action_id_list.index(seq_id)
                ref_sequence.append(ind)
            if len(ref_sequence) == self.seq_length:
                samples[0] = ref_sequence

        return samples

    def save_results(self, res: Result) -> None:
        save_path = self.exp_path()
        log(f'{self.exp_id()} -> Save to {save_path}...')
        os.makedirs(save_path, exist_ok=True)

        # save table of results
        pd_res = self.process_results(res)
        res_path = os.path.join(save_path, 'res.csv')
        pd_res.to_csv(res_path)

        # save execution time
        np.save(os.path.join(save_path, 'exec_time.npy'), np.array(self.exec_time))

        # save config
        save_w_pickle(self.get_config(), save_path, filename='config.pkl')

        # save res
        save_w_pickle(res, save_path, 'res.pkl')
        log(f'{self.exp_id()} -> Saved!')

    def exists(self) -> bool:
        """ Check if experiment already exists """
        save_path = self.exp_path()
        paths_to_check = [
            os.path.join(save_path, 'res.csv'),
            os.path.join(save_path, 'exec_time.npy'),
            os.path.join(save_path, 'config.pkl'),
            os.path.join(save_path, 'res.pkl')
        ]
        return np.all(list(map(lambda p: os.path.exists(p), paths_to_check)))

    @staticmethod
    def get_history_values(res: Result, pop_size: int, n_gen: int) -> Tuple[np.ndarray, np.ndarray]:
        """

        Return an array

        Args:
            res: pymoo result object having `history` field
            pop_size: parameter of NSGA2 (population size)
            n_gen: parameter of NSGA2 (number of generations)

        Returns:
            X: array of inputs (n_gen * pop_size, action_space_size)
            Y: array of obj values (n_gen * pop_size, 2)
        """
        X = np.concatenate([np.stack([res.history[i].pop[j].X for j in range(pop_size)]) for i in range(n_gen)])
        Y = np.concatenate([np.stack([res.history[i].pop[j].F for j in range(pop_size)]) for i in range(n_gen)])
        assert Y.shape == (n_gen * pop_size, 2), (Y.shape, (n_gen * pop_size, 2))
        assert X.shape == (n_gen * pop_size, res.history[0].pop[0].X.shape[0]), (
            X.shape, n_gen * pop_size, res.history[0].pop[0].X.shape[0])
        return X, Y


class NSGA2Problem(Problem):
    """ Pymoo problem formulation for NSGA 2"""

    def __init__(self, n_var: int, action_space: List[Action], design_file: str, mapping: str,
                 library_file: str, abc_binary: str, lut_inputs: int, ref_1: float, ref_2: float):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=len(action_space) - 1, type_var=int,
                         elementwise_evaluation=True)
        self.mapping = mapping
        self.design_file = design_file
        self.library_file = library_file
        self.abc_binary = abc_binary
        self.lut_inputs = lut_inputs
        self.ref_1 = ref_1
        self.ref_2 = ref_2
        self.action_space = action_space

    def _evaluate(self, sequence_inds, out, *args, **kwargs):
        sequence = [self.action_space[ind].act_id for ind in sequence_inds]

        obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=self.design_file, mapping=self.mapping,
                                       library_file=self.library_file, abc_binary=self.abc_binary,
                                       lut_inputs=self.lut_inputs, compute_init_stats=False)
        out['F'] = [obj_1 / self.ref_1, obj_2 / self.ref_2]  # we want to minimize both
