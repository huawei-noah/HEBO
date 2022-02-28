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
from joblib import Parallel, delayed
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling, get_mutation, get_crossover, get_selection
from pymoo.interface import sample
from pymoo.model.problem import Problem
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.optimize import minimize
from pymoo.util.dominator import Dominator
from pymoo.visualization.scatter import Scatter

from core.action_space import Action
from core.algos.common_exp import MultiEADExp
from core.algos.utils import is_pareto_efficient, Res, get_history_values_from_res, get_design_name
from core.sessions.utils import get_design_prop
from utils.utils_misc import time_formatter, log
from utils.utils_save import save_w_pickle, safe_load_w_pickle


class MultiNSGA2Res(Res):
    """ Auxiliary class to mimic pymoo format """

    def __init__(self, X: np.ndarray, F: np.ndarray, history_x: Optional[np.ndarray] = None,
                 history_f: Optional[np.ndarray] = None, mask: np.ndarray = None,
                 full_history_1: np.ndarray = None, full_history_2: np.ndarray = None, valids: np.ndarray = None):
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
        self.valids = valids


class MultiNSGA2Exp(MultiEADExp):
    color = 'grey'
    linestyle = '-'

    method_id: str = 'multi-NSGA2'
    meta_method_id_: str = 'multi-NSGA2'

    @property
    def meta_method_id(self) -> str:
        return self.meta_method_id_

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int,
                 lut_inputs: int,
                 pop_size: int, n_gen: int, eta_mutation: float,
                 eta_cross: int, prob_cross: float, selection: str,
                 ref_abc_seq: Optional[str],
                 use_yosys: bool,
                 n_parallel: int = 1):
        """
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            pop_size: population size for SGA
            n_gen: number of generations
            eta_mutation: eta parameter for int_pm mutation
            eta_cross: eta parameter for crossover
            prob_cross: prob parameter for crossover
            selection: selection type
            seed: reproducibility seed
        """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file,
                         abc_binary=abc_binary, n_parallel=n_parallel,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq, use_yosys=use_yosys)
        self.pop_size = pop_size
        self.seed = seed
        self.n_gen = n_gen
        self.termination = get_termination("n_gen", self.n_gen)

        self.eta_mutation = eta_mutation
        self.eta_cross = eta_cross
        self.prob_cross = prob_cross
        self.selection = selection

        self.sampler = get_sampling("int_lhs")
        self.algo = NSGA2(pop_size=self.pop_size,
                          selection=self.get_selection(),
                          sampling=self.get_samples(),
                          mutation=self.get_mutation(),
                          crossover=self.get_crossover(),
                          eliminate_duplicates=True)

        self.verbose = True

        self.valids_path = os.path.join(self.exp_path(), 'valids.npy')

        # pymoo problem
        self.problem = MultiNSGA2Problem(
            n_var=seq_length,
            action_space=self.action_space,
            design_files=self.design_files,
            mapping=self.mapping,
            library_file=library_file,
            abc_binary=abc_binary,
            lut_inputs=self.lut_inputs,
            refs_1=self.refs_1,
            refs_2=self.refs_2,
            n_parallel=n_parallel,
            use_yosys=self.use_yosys,
            eval_ckpt_root_path=self.eval_ckpt_root_path,
            res_ckpt_path=os.path.join(self.exp_path(), 'res_ckpt.pkl')
        )

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiNSGA2Exp, self).get_config()
        config['pop_size'] = self.pop_size
        config['n_gen'] = self.n_gen
        config['seed'] = self.seed
        config['eta_mutation'] = self.eta_mutation
        config['eta_cross'] = self.eta_cross
        config['prob_cross'] = self.prob_cross
        config['selection'] = self.selection

        return config

    def get_selection(self):
        if self.selection == 'tournament':

            def binary_tournament(pop, P, algorithm, **kwargs):
                if P.shape[1] != 2:
                    raise ValueError("Only implemented for binary tournament!")

                tournament_type = algorithm.tournament_type
                S = np.full(P.shape[0], np.nan)

                for i in range(P.shape[0]):

                    a, b = P[i, 0], P[i, 1]

                    # if at least one solution is infeasible
                    if pop[a].CV > 0.0 or pop[b].CV > 0.0:
                        S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better',
                                       return_random_if_equal=True)

                    # both solutions are feasible
                    else:

                        # if tournament_type == 'comp_by_dom_and_crowding':
                        rel = Dominator.get_relation(pop[a].F, pop[b].F)
                        if rel == 1:
                            S[i] = a
                        elif rel == -1:
                            S[i] = b

                        # elif tournament_type == 'comp_by_rank_and_crowding':
                        #     S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                        #                    method='smaller_is_better')

                        # else:
                        #     raise Exception("Unknown tournament type.")

                        # if rank or domination relation didn't make a decision compare by crowding
                        if np.isnan(S[i]):
                            S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                                           method='larger_is_better', return_random_if_equal=True)

                return S[:, None].astype(int, copy=False)

            return TournamentSelection(func_comp=binary_tournament)
        return get_selection(self.selection)

    def get_mutation(self):
        return get_mutation('int_pm', eta=self.eta_mutation)

    def get_crossover(self):
        return get_crossover('int_sbx', eta=self.eta_cross, prob=self.prob_cross)

    def exp_id(self) -> str:
        return self.get_exp_id(
            pop_size=self.pop_size,
            n_gen=self.n_gen,
            eta_mutation=self.eta_mutation,
            eta_cross=self.eta_cross,
            prob_cross=self.prob_cross,
            selection=self.selection
        )

    @staticmethod
    def get_exp_id(pop_size: int, n_gen: int, eta_mutation: float,
                   eta_cross: int, prob_cross: float, selection: str) -> str:
        exp_id = MultiNSGA2Exp.method_id
        exp_id += f"_pop-{pop_size}"
        exp_id += f"_ngen-{n_gen}"
        exp_id += f"_eta-mute-{eta_mutation:g}"
        exp_id += f"_eta-cross-{eta_cross:g}"
        exp_id += f"_prob-cross-{prob_cross:g}"
        exp_id += f"_selection-{selection}"
        return exp_id

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
        return os.path.join(MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiNSGA2Exp.meta_method_id_,
            mapping=mapping,
            lut_inputs=lut_inputs,
            seq_length=seq_length,
            action_space_id=action_space_id,
            exp_id=exp_id,
            design_files_id=design_files_id,
            ref_abc_seq=ref_abc_seq
        ), str(seed))

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

    def get_samples(self) -> np.ndarray:
        """ Return either an array of initial population obtained with latin hypercube sampling """
        np.random.seed(self.seed)
        samples = sample(self.sampler, n_samples=self.pop_size, n_var=self.seq_length, xl=0,
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
        assert samples.shape == (self.pop_size, self.seq_length), samples.shape
        return samples

    def run(self, verbose: bool = False, visualize: bool = False):
        np.random.seed(self.seed)
        t = time.time()
        self.problem.verbose = verbose
        res = minimize(
            problem=self.problem,
            algorithm=self.algo,
            termination=self.termination,
            seed=self.seed,
            verobse=verbose,
            save_history=True
        )
        self.exec_time = time.time() - t
        if visualize:
            plot = Scatter()
            plot.add(self.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.add(res.F, color="red")
            plot.show()

        objs = np.stack([np.mean(self.problem.full_obj_1_s, 1), np.mean(self.problem.full_obj_2_s, 1)]).T
        assert objs.shape == (self.n_gen * self.pop_size, 2), (objs.shape, self.n_gen, self.pop_size)
        history_x = np.array(self.problem.samples).copy()
        history_f = objs.copy()
        mask_pareto = is_pareto_efficient(objs)
        samples = history_x[mask_pareto]
        objs = objs[mask_pareto]
        assert np.isclose(objs.mean(), res.F.mean()), (objs, res.F)
        res = MultiNSGA2Res(X=samples, F=objs, history_x=history_x, history_f=history_f, mask=mask_pareto,
                            full_history_1=np.array(self.problem.full_obj_1_s),
                            full_history_2=np.array(self.problem.full_obj_2_s), valids=np.array(self.problem.valids))
        self.exec_time = time.time() - t
        if verbose:
            self.log(
                f"Took {time_formatter(self.exec_time)} to optimise {self.designs_group_id} "
                f"-> improvement QoR is {(2 - objs.sum(-1).min()) / 2 * 100:.2f}%")
        return res

    def process_results(self, res: MultiNSGA2Res) -> pd.DataFrame:
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

        pd_res['both'] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values('both')

    def save_results(self, res: MultiNSGA2Res) -> None:
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
    def get_history_values(res: MultiNSGA2Res) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)


class MultiNSGA2Problem(Problem):
    """ Pymoo problem formulation for NSGA 2"""

    def __init__(self, n_var: int, action_space: List[Action], design_files: List[str], mapping: str,
                 library_file: str, abc_binary: str, lut_inputs: int, use_yosys: bool, refs_1: List[float],
                 refs_2: List[float], eval_ckpt_root_path: str, res_ckpt_path: str, verbose: bool = 0,
                 n_parallel: int = 1):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=len(action_space) - 1, type_var=int,
                         elementwise_evaluation=False)
        self.mapping = mapping
        self.design_files = design_files
        self.library_file = library_file
        self.abc_binary = abc_binary
        self.lut_inputs = lut_inputs
        self.refs_1 = refs_1
        self.refs_2 = refs_2
        self.action_space = action_space
        self.n_parallel = n_parallel
        self.use_yosys = use_yosys
        self.verbose = verbose
        self.eval_ckpt_root_path = eval_ckpt_root_path
        self.res_ckpt_path = res_ckpt_path

        self.samples = []
        self.full_obj_1_s = []
        self.full_obj_2_s = []
        self.valids = []

    def get_obj(self, sequence: List[int], design_file: str, ref_1: float, ref_2: float) -> Tuple[
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
                obj_1, obj_2, extra_info = get_design_prop(
                    seq=sequence,
                    design_file=design_file,
                    mapping=self.mapping,
                    library_file=self.library_file,
                    abc_binary=self.abc_binary,
                    lut_inputs=self.lut_inputs,
                    use_yosys=self.use_yosys,
                    compute_init_stats=False
                )
            except CalledProcessError as e:
                if e.args[0] == -6:
                    log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ",
                        "Multi-NSGA2")
                    obj_1 = ref_1
                    obj_2 = ref_2
                    valid = False
                    extra_info = None
                else:
                    raise e

            eval_dic = safe_load_w_pickle(eval_dic_path)
            eval_dic[seq_ind_id] = obj_1, obj_2, extra_info, valid
            save_w_pickle(eval_dic, eval_dic_path)
        return obj_1 / ref_1, obj_2 / ref_2, valid

    def _evaluate(self, sequence_inds: np.ndarray, out, *args, **kwargs):
        """
        Args:
            sequence_inds: array of shape (pop_size, seq_length) containing the sequences to test
        """
        self.samples.extend(sequence_inds)
        offset = len(self.full_obj_1_s)
        self.full_obj_1_s.extend([np.zeros(len(self.design_files)) for _ in range(len(sequence_inds))])
        self.full_obj_2_s.extend([np.zeros(len(self.design_files)) for _ in range(len(sequence_inds))])
        self.valids.extend([np.zeros(len(self.design_files)) for _ in range(len(sequence_inds))])
        for design_file_ind, design_file in enumerate(self.design_files):
            objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
                delayed(self.get_obj)(sequence=sequence_inds[k], design_file=design_file,
                                      ref_1=self.refs_1[design_file_ind],
                                      ref_2=self.refs_2[design_file_ind]) for k in range(len(sequence_inds)))
            for k in range(len(sequence_inds)):
                self.full_obj_1_s[offset + k][design_file_ind] = objs[k][0]
                self.full_obj_2_s[offset + k][design_file_ind] = objs[k][1]
                self.valids[offset + k][design_file_ind] = objs[k][2]
        objs = np.zeros((len(sequence_inds), 2))
        objs[:, 0] = np.mean(self.full_obj_1_s[-len(sequence_inds):], axis=1)
        objs[:, 1] = np.mean(self.full_obj_2_s[-len(sequence_inds):], axis=1)

        assert objs.shape == (len(sequence_inds), 2), objs.shape

        save_w_pickle({
            'samples': self.samples,
            'full_objs_1_s': self.full_obj_1_s,
            'full_objs_2_s': self.full_obj_2_s,
            'valids': self.valids
        }, self.res_ckpt_path)

        out['F'] = objs
        if self.verbose:
            self.log(f"Evaluated {len(sequence_inds)} sequences.")

    def log(self, msg: str, end=None) -> None:
        log(msg,
            header=f"NSGA2 - {' '.join(map(lambda path: os.path.basename(path).split('.')[0], self.design_files))}",
            end=end)
