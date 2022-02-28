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
from subprocess import CalledProcessError
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga
from joblib import Parallel, delayed
from pymoo.factory import get_sampling
from pymoo.interface import sample

from core.algos.common_exp import MultiEADExp, Checkpoint
from core.algos.genetic.sga.utils_sga import TooManyEvaluationsError
from core.algos.utils import is_pareto_efficient, Res, get_history_values_from_res, get_design_name
from core.sessions.utils import get_design_prop
from utils.utils_misc import time_formatter, log
from utils.utils_save import save_w_pickle, load_w_pickle, safe_load_w_pickle


class MultiSGARes(Res):
    """ Auxiliary class to mimic pymoo format """

    def __init__(self, X: np.ndarray, F: np.ndarray, history_x: Optional[np.ndarray] = None,
                 history_f: Optional[np.ndarray] = None, mask: np.ndarray = None,
                 full_history_1: np.ndarray = None, full_history_2: np.ndarray = None):
        """

        Args:
            mask: binary array indicating whether each element belongs to best points
            full_history_1: obj1 for all designs and seeds
            full_history_2: obj2 for all designs and seeds
            X: best points (pareto front if multi-objective)
            F: function values (shape: (n_points, n_obj_functions)
            history_x: all
        """
        super().__init__(X, F, history_x, history_f)
        self.full_history_1 = full_history_1
        self.full_history_2 = full_history_2
        self.mask = mask


def obj_both(ratio_1, ratio_2):
    return ratio_1 + ratio_2


def obj_level(ratio_1, ratio_2):
    return ratio_2


def obj_lut(ratio_1, ratio_2):
    return ratio_1


def obj_min_improvements(ratio_1, ratio_2):
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


class MultiSGAExp(MultiEADExp):
    color = 'green'
    linestyle = '-'

    method_id: str = 'multi-SGA'
    meta_method_id_: str = 'multi-SGA'

    @property
    def meta_method_id(self) -> str:
        return self.meta_method_id_

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int,
                 lut_inputs: int,
                 use_yosys: bool,
                 pop_size: int, n_total_evals: int, parents_portion: float, mutation_probability: float,
                 elit_ration: float, crossover_probability: float, crossover_type: str, objective: str,
                 n_parallel: int = 1,
                 ref_abc_seq: Optional[str] = None):
        """
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            use_yosys: whether to use yosys-abc or abc_py
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            pop_size: population size for SGA
            n_total_evals: number of sequences to evaluate
            mutation_probability: determines the chance of each gene in sequence to be replaced by a random value
            seed: reproducibility seed
            objective: quantity to optimize, either lut, level, or both
        """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, use_yosys=use_yosys,
                         abc_binary=abc_binary, n_parallel=n_parallel,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq)
        self.pop_size = pop_size
        self.seed = seed
        self.n_total_evals = n_total_evals

        self.sampler = get_sampling("int_lhs")

        self.parents_portion = parents_portion
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elit_ration = elit_ration

        self.samples = np.zeros((self.n_total_evals, self.seq_length),
                                dtype=int)  # obj_1 for each sample for each design file
        self.full_obj_1_s = np.zeros(
            (self.n_total_evals, len(self.design_files)))  # obj_1 for each sample for each design file
        self.full_obj_2_s = np.zeros(
            (self.n_total_evals, len(self.design_files)))  # obj_2 for each sample for each design file
        self.valids = np.zeros((self.n_total_evals, len(self.design_files)))
        self.current_sample_ind = 0

        self.obj_1_s: List[float] = []
        self.obj_2_s: List[float] = []

        self.varbound = np.array([[0, len(self.action_space) - 1]] * self.seq_length)

        self.verbose = True
        assert objective in ['both', 'lut', 'level'], objective
        self.objective = objective

        if self.objective == 'both':
            self.objective_function = obj_both
        elif self.objective == 'lut':
            self.objective_function = obj_lut
        elif self.objective == 'level':
            self.objective_function = obj_level
        elif self.objective == 'min_improvements':
            self.objective_function = obj_min_improvements  # improvement is 1 - ratio
        else:
            raise ValueError(self.objective)

        self.valids_path = os.path.join(self.exp_path(), 'valids.npy')

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiSGAExp, self).get_config()
        config['pop_size'] = self.pop_size
        config['n_total_evals'] = self.n_total_evals
        config['seed'] = self.seed
        config['parents_portion'] = self.parents_portion
        config['crossover_probability'] = self.crossover_probability
        config['crossover_type'] = self.crossover_type
        config['elit_ration'] = self.elit_ration
        config['mutation_probability'] = self.mutation_probability
        config['objective'] = self.objective
        config['use_yosys'] = self.use_yosys
        return config

    @property
    def max_num_iteration(self):
        """ n_total_evals = pop_size + (1 - parents_portion) * pop_size * max_num_iterations """
        return np.ceil((self.n_total_evals - self.pop_size) / ((1 - self.parents_portion) * self.pop_size))

    def exp_id(self):
        return self.get_exp_id(pop_size=self.pop_size, n_total_evals=self.n_total_evals,
                               parents_portion=self.parents_portion, mutation_probability=self.mutation_probability,
                               elit_ration=self.elit_ration, crossover_probability=self.crossover_probability,
                               crossover_type=self.crossover_type, objective=self.objective, use_yosys=self.use_yosys)

    @staticmethod
    def get_exp_id(pop_size: int, n_total_evals: int, parents_portion: float, mutation_probability: float,
                   elit_ration: float, crossover_probability: float, crossover_type: str, objective: str,
                   use_yosys: bool) -> str:
        exp_id = f'multi-SGA_pop-{pop_size}_tot-evals-{n_total_evals}'
        if parents_portion != .3:
            exp_id += f'_parents-portion-{parents_portion:g}'
        if mutation_probability != .1:
            exp_id += f'_mutation-{mutation_probability:g}'
        if elit_ration != .01:
            exp_id += f'_elit-{elit_ration}'
        if crossover_probability != .5:
            exp_id += f'_cross-p-{crossover_probability}'
        if crossover_type != 'uniform':
            exp_id += f'_cross-t-{crossover_type}'
        exp_id += f'_obj-{objective}'
        if use_yosys:
            exp_id += '_yosys'
        return exp_id

    def exp_path(self) -> str:
        return self.get_exp_path(
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_abc_seq=self.ref_abc_seq,
            seed=self.seed
        )

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
        return os.path.join(MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiSGAExp.meta_method_id_,
            mapping=mapping,
            lut_inputs=lut_inputs,
            seq_length=seq_length,
            action_space_id=action_space_id,
            exp_id=exp_id,
            design_files_id=design_files_id,
            ref_abc_seq=ref_abc_seq
        ), str(seed))

    def get_samples(self, n_samples: int) -> np.ndarray:
        """ Return either an array of initial population obtained with latin hypercube sampling """
        samples = sample(self.sampler, n_samples=n_samples, n_var=self.seq_length, xl=0,
                         xu=len(self.action_space) - 1)

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
        assert samples.shape == (n_samples, self.seq_length), samples.shape
        return samples

    def get_obj(self, sequence: np.ndarray, design_file: str, ref_1: float, ref_2: float) -> Tuple[
        float, float, bool]:
        """ Return either area and delay or lut and levels """
        seq_ind_id = "-".join(sequence.astype(str))
        design_id = get_design_name(design_file)
        eval_dic_path = os.path.join(self.eval_ckpt_root_path, design_id, 'eval.pkl')
        if not os.path.exists(eval_dic_path):
            os.makedirs(os.path.dirname(eval_dic_path), exist_ok=True)
            save_w_pickle({}, eval_dic_path)
        eval_dic = safe_load_w_pickle(eval_dic_path)

        if seq_ind_id in eval_dic:
            obj_1, obj_2, extra_info, valid = eval_dic[seq_ind_id]
        else:
            sequence = [(self.action_space[ind].act_id if not self.use_yosys else self.action_space[ind].act_str) for
                        ind in
                        sequence]

            valid = True
            try:
                obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=design_file, mapping=self.mapping,
                                                           use_yosys=self.use_yosys,
                                                           library_file=self.library_file, abc_binary=self.abc_binary,
                                                           lut_inputs=self.lut_inputs, compute_init_stats=False)
            except CalledProcessError as e:
                if e.args[0] == -6:
                    self.log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ")
                    obj_1 = ref_1
                    obj_2 = ref_2
                    valid = False
                    extra_info = None
                else:
                    raise e
            eval_dic = safe_load_w_pickle(eval_dic_path, n_trials=5, time_sleep=2 + np.random.random() * 3)
            eval_dic[seq_ind_id] = obj_1, obj_2, extra_info, valid
            save_w_pickle(eval_dic, eval_dic_path)
        return obj_1 / ref_1, obj_2 / ref_2, valid

    def run(self, n_parallel: int = 1, verbose: bool = True, overwrite: bool = False) -> MultiSGARes:
        t = time.time()
        self.verbose = verbose
        np.random.seed(self.seed * 1234 + 1)

        model = ga(lambda X: self.obj_func(X, n_parallel=n_parallel), dimension=self.seq_length,
                   variable_type='int',
                   variable_boundaries=self.varbound,
                   variable_type_mixed=None,
                   function_timeout=1e9,
                   algorithm_parameters={'max_num_iteration': self.max_num_iteration,
                                         'population_size': self.pop_size,
                                         'mutation_probability': self.mutation_probability,
                                         'elit_ratio': self.elit_ration,
                                         'crossover_probability': self.crossover_probability,
                                         'parents_portion': self.parents_portion,
                                         'crossover_type': self.crossover_type,
                                         'selection_type': 'roulette',
                                         'max_iteration_without_improv': None})

        start_generation = {'variables': self.get_samples(n_samples=self.pop_size), 'scores': None}
        try:
            model.run(
                set_function=model.f,
                no_plot=True,
                start_generation=start_generation,
                seed=self.seed, disable_progress_bar=not verbose,
            )
        except TooManyEvaluationsError:
            self.exec_time = time.time() - t

        if self.current_sample_ind < self.n_total_evals:
            samples = self.get_samples(n_samples=self.n_total_evals - self.current_sample_ind)
            self.log(f"Complete run with {len(samples)} random trials")
            for sample_ in samples:
                self.obj_func(X=sample_, n_parallel=n_parallel)

        objs = np.stack([self.full_obj_1_s.mean(1), self.full_obj_2_s.mean(1)]).T
        assert objs.shape == (self.n_total_evals, 2), objs.shape
        if self.objective == 'both':
            # pareto
            mask = is_pareto_efficient(objs)
        elif self.objective == 'level':
            mask = objs[:, 0] == objs[:, 0].min()
        elif self.objective == 'lut':
            mask = objs[:, 1] == objs[:, 1].min()
        elif self.objective == 'min_improvements':
            aux_objs = np.array([
                np.mean([
                    self.objective_function(o1, o2) for (o1, o2) in zip(self.full_obj_1_s[sample_ind],
                                                                        self.full_obj_2_s[sample_ind])])
                for sample_ind in range(len(self.full_obj_1_s))
            ])
            mask = aux_objs == aux_objs.min()
        else:
            raise ValueError(self.objective)
        history_x = self.samples.copy()
        history_f = objs.copy()
        samples = self.samples[mask]
        objs = objs[mask]
        res = MultiSGARes(X=samples, F=objs, history_x=history_x, history_f=history_f, mask=mask,
                          full_history_1=self.full_obj_1_s, full_history_2=self.full_obj_2_s)
        self.exec_time = time.time() - t
        if verbose:
            self.log(
                f"{self.seed}. Took {time_formatter(self.exec_time)} to optimise {self.designs_group_id} "
                f"-> improvement {self.objective} is {objs.sum(-1).min() * 100:.2f}%")
        return res

    def process_results(self, res: MultiSGARes) -> pd.DataFrame:
        seq_id = []
        obj_1 = []
        ratio_1 = []
        obj_2 = []
        ratio_2 = []
        for seq_ind, func_value in zip(res.X, res.F):
            seq_id.append(' | '.join([self.action_space[ind].act_id for ind in seq_ind]))
            ratio_1.append(func_value[0])
            ratio_2.append(func_value[1])
            # obj_1.append(ratio_1[-1] * self.ref_1)
            # obj_2.append(ratio_2[-1] * self.ref_2)
        pd_res = pd.DataFrame()
        pd_res['seq_id'] = seq_id

        # pd_res[self.obj1_id] = obj_1
        # pd_res[self.obj2_id] = obj_2

        pd_res['ratio ' + self.obj1_id] = ratio_1
        pd_res['ratio ' + self.obj2_id] = ratio_2

        pd_res[self.objective] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values(self.objective)

    def obj_func(self, X: np.ndarray, n_parallel: int) -> np.ndarray:
        """ Objective function to minimize """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.current_sample_ind + len(X) >= self.n_total_evals:
            raise TooManyEvaluationsError()
        X = X.astype(int)
        self.samples[self.current_sample_ind:self.current_sample_ind + len(X)] = X
        for design_file_ind, design_file in enumerate(self.design_files):
            objs = Parallel(n_jobs=n_parallel, backend="multiprocessing")(
                delayed(self.get_obj)(sequence=X[k], design_file=design_file,
                                      ref_1=self.refs_1[design_file_ind],
                                      ref_2=self.refs_2[design_file_ind]) for k in range(len(X)))
            for k in range(len(X)):
                self.full_obj_1_s[self.current_sample_ind + k] = objs[k][0]
                self.full_obj_2_s[self.current_sample_ind + k] = objs[k][1]
                self.valids[self.current_sample_ind + k] = objs[k][2]

        np.save(self.valids_path, self.valids)
        save_w_pickle(
            Checkpoint(samples=self.samples[:self.current_sample_ind + len(X)], full_objs_1=self.full_obj_1_s,
                       full_objs_2=self.full_obj_2_s),
            path=self.exp_path(),
            filename='ckpt.pkl'
        )

        real_obj_mean = np.array([np.mean(
            [
                self.objective_function(o1, o2) for (o1, o2) in zip(self.full_obj_1_s[self.current_sample_ind + k],
                                                                    self.full_obj_2_s[self.current_sample_ind + k])
            ]) for k in range(len(X))]
        )

        self.current_sample_ind += len(X)

        real_obj_min_mean = np.min([
            np.mean([
                self.objective_function(o1, o2) for (o1, o2) in zip(self.full_obj_1_s[sample_ind],
                                                                    self.full_obj_2_s[sample_ind])])
            for sample_ind in range(self.current_sample_ind)
        ])

        self.log(
            f"{self.designs_group_id} {self.seed} -> {self.current_sample_ind} / {self.n_total_evals}: "
            f"Best {self.objective} -> {real_obj_min_mean * 100:.2f}%")

        return real_obj_mean

    def save_results(self, res: MultiSGARes) -> None:
        save_path = self.exp_path()
        self.log(f'{self.exp_id()} -> Save to {save_path}...')
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
        self.log(f'{self.exp_id()} -> Saved!')

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

    def log(self, msg: str, end=None) -> None:
        if self.verbose:
            log(msg, header=self.method_id, end=end)

    @staticmethod
    def get_history_values(res: MultiSGARes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)
