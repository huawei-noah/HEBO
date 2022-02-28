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

import numpy as np
import os
import pandas as pd
import shutil
import time
import torch
from joblib import Parallel, delayed
from subprocess import CalledProcessError
from typing import Dict, Any, Optional, Tuple

from core.algos.common_exp import Checkpoint, MultiseqEADExp
from core.algos.utils import Res, get_history_values_from_res, get_design_name
from core.sessions.utils import get_design_prop
from resources.casmopolitan.bo.optimizer import Optimizer
from utils.utils_misc import log, time_formatter
from utils.utils_save import save_w_pickle, load_w_pickle


class MultiseqBoilsRes(Res):
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
        assert full_history_1.shape == full_history_2.shape
        assert full_history_1.ndim == full_history_2.ndim == 3, \
            full_history_1.ndim  # n_seqs x n_tested_seqs x n_circuits
        self.full_history_1 = full_history_1
        self.full_history_2 = full_history_2
        self.mask = mask
        self.valids = valids


def obj_both(ratio_1, ratio_2):
    return ratio_1 + ratio_2


def obj_level(ratio_1, ratio_2):
    return ratio_2


def obj_lut(ratio_1, ratio_2):
    return ratio_1


def obj_min_improvements(ratio_1, ratio_2):
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


class MultiseqBoilsExp(MultiseqEADExp):
    """ Class associated to BOiLS to solve QoR minimization: https://arxiv.org/abs/2111.06178 """

    color = 'cyan'

    method_id: str = 'BOiLS'
    meta_method_id: str = 'BO'

    def __init__(self, designs_group_id: str, seq_length: int, n_universal_seqs: int, mapping: str,
                 action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int, n_initial: int, standardise: bool, ard: bool, acq: str,
                 lut_inputs: int,
                 ref_abc_seq: Optional[str], objective: str, overwrite: bool, n_parallel: int = 1):
        """
        Args:
            designs_group_id: id of the group of designs
            seq_length: length of the optimal sequence to find
            n_universal_seqs: number of sequences
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            seed: reproducibility seed
            n_initial: number of initial points to test before building first surrogate model
            standardise: whether to standardise the ys for the GP fit
            ard: whether to enable automatic relevance determination
            acq: choice of the acquisition function.
            objective: quantity to optimize, either lut, level, both or min_improvements
       """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, n_universal_seqs=n_universal_seqs,
                         mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, abc_binary=abc_binary,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq)
        self.seed = seed

        self.n_initial = n_initial
        self.standardise = standardise
        self.ard = ard
        self.acq = acq

        assert objective in ['both', 'lut', 'level', 'min_improvements'], objective

        self.objective = objective

        if self.objective == 'both':
            self.objective_function = obj_both
        elif self.objective == 'lut':
            self.objective_function = obj_lut
        elif self.objective == 'level':
            self.objective_function = obj_level
        elif self.objective == 'min_improvements':
            self.objective_function = obj_min_improvements
        else:
            raise ValueError(self.objective)

        self.ref_time = time.time()

        self.samples_X = []
        self.objs = []
        self.full_obj_1_s = []
        self.full_obj_2_s = []
        self.valids = []

        self.n_evals = 0
        self.n_parallel = n_parallel

        self.valids_path = os.path.join(self.exp_path(), 'valids.npy')

        self.playground: str = os.path.join(self.exp_path(), 'playground')
        if overwrite:
            self.log(f"Overwrite: remove {self.playground}")
            shutil.rmtree(self.playground, ignore_errors=True)
            if os.path.exists(os.path.join(self.exp_path(), 'ckpt.pkl')):
                os.remove(os.path.join(self.exp_path(), 'ckpt.pkl'))
        os.makedirs(self.playground, exist_ok=True)

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiseqBoilsExp, self).get_config()
        config['seed'] = self.seed
        config['n_initial'] = self.n_initial
        config['ard'] = self.ard
        config['acq'] = self.acq
        config['standardise'] = self.standardise
        config['objective'] = self.objective
        return config

    @staticmethod
    def get_exp_id(standardise: bool, n_initial: int, objective: str, ard: bool, acq: str) -> str:
        exp_id = MultiseqBoilsExp.method_id
        if standardise:
            exp_id += f"_std"
        exp_id += f"_init-{n_initial}"
        exp_id += f"_obj-{objective}"
        exp_id += f"_acq-{acq}"
        if ard:
            exp_id += f"_ard"
        return exp_id

    def exp_id(self) -> str:
        return self.get_exp_id(
            standardise=self.standardise,
            n_initial=self.n_initial,
            objective=self.objective,
            acq=self.acq,
            ard=self.ard
        )

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int, n_universal_seqs: int):
        return os.path.join(MultiseqEADExp.get_exp_path_aux(
            meta_method_id=MultiseqBoilsExp.meta_method_id,
            mapping=mapping,
            lut_inputs=lut_inputs,
            seq_length=seq_length,
            n_universal_seqs=n_universal_seqs,
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
            seed=self.seed,
            n_universal_seqs=self.n_universal_seqs
        )

    def run(self, device: Optional[int], n_total_evals: int, verbose: bool = False):
        n_categories = np.array([self.action_space_length] * self.seq_length * self.n_universal_seqs)
        problem_type = 'categorical'
        noise_variance = None
        kernel_type = 'mixed' if problem_type == 'mixed' else 'transformed_overlap'
        kwargs = {
            'length_max_discrete': self.seq_length * self.n_universal_seqs,
            'length_init_discrete': max(1, int(2 * self.seq_length * self.n_universal_seqs / 3)),
            'device': device
        }

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        ckpt_path = os.path.join(self.exp_path(), 'ckpt.pkl')
        if os.path.exists(ckpt_path):
            ckpt = load_w_pickle(os.path.dirname(ckpt_path), os.path.basename(ckpt_path))
            self.full_obj_1_s.extend(list(ckpt.full_objs_1))
            self.full_obj_2_s.extend(list(ckpt.full_objs_2))
            self.samples_X.extend(list(ckpt.samples))

        optim = Optimizer(
            config=n_categories,
            n_init=self.n_initial,
            use_ard=self.ard,
            acq=self.acq,
            kernel_type=kernel_type,
            noise_variance=noise_variance,
            standardise=self.standardise,
            **kwargs
        )

        for i in range(n_total_evals):
            if len(self.samples_X) > i:
                x_next = self.samples_X[i].reshape(1, -1)
                if i < self.n_initial:
                    # purgate initials
                    optim.suggest(n_suggestions=1)
            else:
                x_next = optim.suggest(n_suggestions=1)
            y_next: np.ndarray = np.array([self.evaluate(x_next, iter=i)])
            optim.observe(x_next, y_next)

        return self.build_res(verbose=verbose)

    def get_obj(self, sequences: np.ndarray, design_file: str, ref_1: float, ref_2: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            sequences: sequences to evaluate (number of sequences corresponds to number of desired universal sequences)
            design_file: file where circuit is stored
            ref_1: reference for first objective for this circuit
            ref_2: reference for second objective for this circuit

        Returns:
            ratios_1: array (n_universal_seqs, 1) containing obj_1 / ref_1 for each sequence
            ratios_2: array (n_universal_seqs, 1) containing obj_2 / ref_2 for each sequence
            valids: array (n_universal_seqs, 1) containing for entry `i` a boolean corresponding to validity of
                evaluation of the i'th sequence
        """
        ratios_1 = []
        ratios_2 = []
        valids = []
        for sequence in sequences:
            sequence = [(self.action_space[ind].act_id if not self.use_yosys else self.action_space[ind].act_str) for
                        ind in sequence]
            sequence_id = ' '.join(map(str, sequence))
            save_file = os.path.join(self.playground, get_design_name(design_file), 'seq_to_func_dic.pkl')
            seq_to_func_dic: Dict[str, Tuple[float, float, bool]] = {}
            if not os.path.exists(save_file):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_w_pickle(seq_to_func_dic, os.path.dirname(save_file), os.path.basename(save_file))
            else:
                seq_to_func_dic = load_w_pickle(os.path.dirname(save_file), os.path.basename(save_file))

            if sequence_id not in seq_to_func_dic:
                valid = True
                try:
                    self.log(f"{self.n_evals}. Evaluate {sequence_id} for {get_design_name(design_file)} ")
                    obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=design_file,
                                                               mapping=self.mapping,
                                                               library_file=self.library_file,
                                                               abc_binary=self.abc_binary,
                                                               lut_inputs=self.lut_inputs, compute_init_stats=False)
                except CalledProcessError as e:
                    if e.args[0] == -6:
                        self.log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ")
                        obj_1 = ref_1
                        obj_2 = ref_2
                        valid = False
                    else:
                        raise e
                ratio_1 = obj_1 / ref_1
                ratio_2 = obj_2 / ref_2
                seq_to_func_dic[sequence_id] = ratio_1, ratio_2, valid
                save_w_pickle(seq_to_func_dic, os.path.dirname(save_file), os.path.basename(save_file))
            else:
                self.log(f"{self.n_evals}. Already computed {sequence_id} for {get_design_name(design_file)}...")
                ratio_1, ratio_2, valid = seq_to_func_dic[sequence_id]
            ratios_1.append(ratio_1)
            ratios_2.append(ratio_2)
            valids.append(valid)

        return np.array(ratios_1), np.array(ratios_2), np.array(valids)

    def evaluate(self, x: np.ndarray, iter: int) -> float:
        """

        Args:
            x: new point to evaluate (contains n_universal_seqs actions)
        """
        self.n_evals += 1
        # self.log(f"{self.n_evals:3d}. Evaluate sequence {x} on design {self.design_name}: ", end="")
        X = x.astype(int).reshape(self.n_universal_seqs, self.seq_length)

        objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
            delayed(self.get_obj)(sequences=X, design_file=self.design_files[k], ref_1=self.refs_1[k],
                                  ref_2=self.refs_2[k]) for k in range(len(self.design_files)))

        if len(self.full_obj_1_s) <= iter:
            # new inputs
            self.samples_X.append(X)
            self.full_obj_1_s.append([o[0] for o in objs])
            self.full_obj_2_s.append([o[1] for o in objs])

            save_w_pickle(
                Checkpoint(samples=np.array(self.samples_X), full_objs_1=np.array(self.full_obj_1_s),
                           full_objs_2=np.array(self.full_obj_2_s)),
                path=self.exp_path(),
                filename='ckpt.pkl'
            )
        self.valids.append([o[2] for o in objs])
        assert np.array(self.full_obj_1_s).shape[1:] == np.array(self.full_obj_2_s).shape[1:] == (
            len(self.design_files), self.n_universal_seqs), (np.array(self.full_obj_1_s).shape,
                                                             len(self.design_files), self.n_universal_seqs)
        objs = np.zeros((len(self.design_files), self.n_universal_seqs))
        for i in range(len(self.design_files)):
            for j in range(self.n_universal_seqs):
                objs[i, j] = self.objective_function(self.full_obj_1_s[-1][i][j], self.full_obj_2_s[-1][i][j])
        obj = np.mean(np.min(objs, 1))
        self.objs.append(obj)
        return obj

    def build_res(self, verbose: bool = False) -> MultiseqBoilsRes:
        self.full_obj_1_s = np.array(self.full_obj_1_s)
        self.full_obj_2_s = np.array(self.full_obj_2_s)
        self.objs = np.array(self.objs)
        assert self.objs.shape == (self.full_obj_1_s.shape[0],), self.objs.shape
        assert self.full_obj_1_s.shape == self.full_obj_2_s.shape == \
               (len(self.samples_X), len(self.design_files), self.n_universal_seqs)
        history_x = np.array(self.samples_X).copy()
        history_f = self.objs.copy()
        mask = self.objs == self.objs.min()
        samples = history_x[mask]
        objs = self.objs[mask]
        res = MultiseqBoilsRes(X=samples, F=objs, history_x=history_x, history_f=history_f, mask=mask,
                               full_history_1=np.array(self.full_obj_1_s),
                               full_history_2=np.array(self.full_obj_2_s), valids=np.array(self.valids))
        self.exec_time = time.time() - self.ref_time
        if verbose:
            self.log(
                f"Took {time_formatter(self.exec_time)} to optimise {self.designs_group_id} "
                f"-> improvement QoR is {(2 - objs.sum(-1).min()) * 50:.2f}")
        return res

    def log(self, msg: str, end=None) -> None:
        log(msg, header=self.method_id, end=end)

    def process_results(self, res: MultiseqBoilsRes) -> pd.DataFrame:
        raise NotImplementedError()
        # seq_id = []
        # obj_1 = []
        # ratio_1 = []
        # obj_2 = []
        # ratio_2 = []
        # for seq_ind, func_value in zip(res.X, res.F):
        #     seq_id.append(' | '.join([self.action_space[ind].act_id for ind in seq_ind]))
        #     ratio_1.append(func_value[0])
        #     ratio_2.append(func_value[1])
        #     # obj_1.append(ratio_1[-1] * self.ref_1)
        #     # obj_2.append(ratio_2[-1] * self.ref_2)
        # pd_res = pd.DataFrame()
        # pd_res['seq_id'] = seq_id
        #
        # # pd_res[self.obj1_id] = obj_1
        # # pd_res[self.obj2_id] = obj_2
        #
        # pd_res['ratio ' + self.obj1_id] = ratio_1
        # pd_res['ratio ' + self.obj2_id] = ratio_2
        #
        # pd_res['both'] = np.array(ratio_1) + np.array(ratio_2)
        #
        # return pd_res.sort_values('both')

    def save_results(self, res: MultiseqBoilsRes) -> None:
        save_path = self.exp_path()
        self.log(f'{self.exp_id()} -> Save to {save_path}...')
        os.makedirs(save_path, exist_ok=True)

        # save table of results
        # pd_res = self.process_results(res)
        # res_path = os.path.join(save_path, 'res.csv')
        # pd_res.to_csv(res_path)

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
            # os.path.join(save_path, 'res.csv'),
            os.path.join(save_path, 'exec_time.npy'),
            os.path.join(save_path, 'config.pkl'),
            os.path.join(save_path, 'res.pkl')
        ]
        return np.all(list(map(lambda p: os.path.exists(p), paths_to_check)))

    @staticmethod
    def get_history_values(res: MultiseqBoilsRes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)
