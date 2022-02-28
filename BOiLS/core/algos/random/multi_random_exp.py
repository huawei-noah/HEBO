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

import itertools

import numpy as np
import os
import pandas as pd
import time
from joblib import Parallel, delayed
from pymoo.factory import get_sampling
from pymoo.interface import sample
from subprocess import CalledProcessError
from typing import Optional, Dict, Any, Tuple, List

from core.algos.common_exp import MultiEADExp, Checkpoint
from core.algos.utils import is_pareto_efficient, Res, get_history_values_from_res, get_design_name
from core.sessions.utils import get_design_prop
from utils.utils_misc import time_formatter, log
from utils.utils_save import save_w_pickle, load_w_pickle


class MultiRandomRes(Res):
    """ Auxiliary class to mimic pymoo format """

    def __init__(self, X: np.ndarray, F: np.ndarray, history_x: Optional[np.ndarray] = None,
                 history_f: Optional[np.ndarray] = None, mask_pareto: np.ndarray = None,
                 full_history_1: np.ndarray = None, full_history_2: np.ndarray = None):
        """

        Args:
            mask_pareto: binary array indicating whether each element belongs to pareto front
            full_history_1: obj1 for all designs and seeds
            full_history_2: obj2 for all designs and seeds
            X: best points (pareto front if multi-objective)
            F: function values (shape: (n_points, n_obj_functions)
            history_x: all
        """
        super().__init__(X, F, history_x, history_f)
        self.full_history_1 = full_history_1
        self.full_history_2 = full_history_2
        self.mask_pareto = mask_pareto


class MultiRandomExp(MultiEADExp):
    color = 'green'
    linestyle = '-'
    meta_method_id_: str = 'multi-RS'

    @property
    def meta_method_id(self) -> str:
        return self.meta_method_id_

    @property
    def method_id(self) -> str:
        return self.get_method_id(random_sampling_id=self.random_sampling_id)

    @staticmethod
    def get_method_id(random_sampling_id: str):
        return f'multi-RS ({random_sampling_id})'

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int, n_trials: int, use_yosys: bool,
                 lut_inputs: int,
                 random_sampling_id: str, n_parallel: int = 1,
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
        """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, use_yosys=use_yosys,
                         abc_binary=abc_binary, n_parallel=n_parallel,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq)
        self.n_trials = n_trials
        self.seed = seed

        self.random_sampling_id = random_sampling_id
        if self.random_sampling_id == 'latin-hypercube':
            self.sampler = get_sampling("int_lhs")
        elif self.random_sampling_id == 'random':
            self.sampler = get_sampling("int_random")
        else:
            raise ValueError(self.sampler)
        self.res = None

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiRandomExp, self).get_config()
        config['random_sampling_id'] = self.random_sampling_id
        config['n_trials'] = self.n_trials
        config['seed'] = self.seed
        return config

    def exp_id(self) -> str:
        return self.get_exp_id(
            random_sampling_id=self.random_sampling_id,
            n_trials=self.n_trials,
            use_yosys=self.use_yosys
        )

    @staticmethod
    def get_exp_id(random_sampling_id: str, n_trials: int, use_yosys: bool) -> str:
        exp_id = f'{random_sampling_id}-{n_trials}'
        if use_yosys:
            exp_id += '_yosys'
        return exp_id

    def exp_path(self) -> str:
        return os.path.join(super(MultiRandomExp, self).exp_path(), str(self.seed))

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
        return os.path.join(MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiRandomExp.meta_method_id_,
            mapping=mapping,
            lut_inputs=lut_inputs,
            seq_length=seq_length,
            action_space_id=action_space_id,
            exp_id=exp_id,
            design_files_id=design_files_id,
            ref_abc_seq=ref_abc_seq
        ), str(seed))

    def get_obj(self, sequence: List[int], design_file: str, ref_1: float, ref_2: float, compute_init_stats: bool,
                ind_design: int, ind_sample: int) -> \
            Tuple[float, float, bool]:
        """ Return either area and delay or lut and levels """
        sequence = [(self.action_space[ind].act_id if not self.use_yosys else self.action_space[ind].act_str) for ind in
                    sequence]

        valid = True
        try:
            self.log(f"Start: {design_file} ({ind_design}) -- {sequence} ({ind_sample})")
            obj_1, obj_2, extra_info = get_design_prop(
                seq=sequence, design_file=design_file, mapping=self.mapping,
                library_file=self.library_file, abc_binary=self.abc_binary,
                lut_inputs=self.lut_inputs, compute_init_stats=compute_init_stats, verbose=False,
                use_yosys=self.use_yosys
            )
            self.log(f"End: {design_file} ({ind_design}) -- {sequence} ({ind_sample})")
        except CalledProcessError as e:
            if e.args[0] == -6:
                self.log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ")
                obj_1 = ref_1
                obj_2 = ref_2
                valid = False
            else:
                raise e
        return obj_1 / ref_1, obj_2 / ref_2, valid

    def run(self, n_parallel: int = 1, verbose: bool = True, overwrite: bool = False) -> MultiRandomRes:
        np.random.seed(self.seed)
        t = time.time()
        samples = self.get_samples()
        full_obj_1_s = np.zeros((len(samples), len(self.design_files)))  # obj_1 for each sample for each design file
        full_obj_2_s = np.zeros((len(samples), len(self.design_files)))  # obj_2 for each sample for each design file
        valids = np.zeros((len(samples), len(self.design_files)))
        i = 0

        valids_path = os.path.join(self.exp_path(), 'valids.npy')
        if not overwrite and os.path.exists(os.path.join(self.exp_path(), 'ckpt.pkl')):
            ckpt: Checkpoint = load_w_pickle(self.exp_path(), 'ckpt.pkl')
            i = len(ckpt.samples)
            samples[:i] = ckpt.samples
            full_obj_1_s = ckpt.full_objs_1
            full_obj_2_s = ckpt.full_objs_2
            valids[:i] = 1

        assert samples.shape == (self.n_trials, self.seq_length), samples.shape
        while i < len(samples):

            ind_prod = list(
                itertools.product(range(len(self.design_files)), range(i, min(i + n_parallel, len(samples)))))
            objs = Parallel(n_jobs=n_parallel, backend="multiprocessing")(
                delayed(self.get_obj)(sequence=samples[ind_sample], design_file=self.design_files[ind_design],
                                      ref_1=self.refs_1[ind_design],
                                      ref_2=self.refs_2[ind_design],
                                      compute_init_stats=False, ind_design=ind_design, ind_sample=ind_sample)
                for ind_design, ind_sample in ind_prod
            )
            for j in range(len(ind_prod)):
                ind_design, ind_sample = ind_prod[j]
                full_obj_1_s[ind_sample, ind_design] = objs[j][0]
                full_obj_2_s[ind_sample, ind_design] = objs[j][1]
                valids[ind_sample, ind_design] = int(objs[j][2])

            np.save(valids_path, valids)
            save_w_pickle(
                Checkpoint(samples=samples[:i + n_parallel], full_objs_1=full_obj_1_s, full_objs_2=full_obj_2_s),
                path=self.exp_path(),
                filename='ckpt.pkl'
            )

            i += n_parallel
            QoR_mean_min = (full_obj_1_s[:i].mean(-1) + full_obj_2_s[:i].mean(-1)).min()
            self.log(
                f"{self.designs_group_id} -> {min(i, len(samples))} / {len(samples)}: "
                f"Best QoR -> {((2 - QoR_mean_min) / 2) * 100:.2f}%")

        objs = np.stack([full_obj_1_s.mean(1), full_obj_2_s.mean(1)]).T
        assert objs.shape == (self.n_trials, 2), objs.shape
        mask_pareto = is_pareto_efficient(objs)
        history_x = samples.copy()
        history_f = objs.copy()
        samples = samples[mask_pareto]
        objs = objs[mask_pareto]
        res = MultiRandomRes(X=samples, F=objs, history_x=history_x, history_f=history_f, mask_pareto=mask_pareto,
                             full_history_1=full_obj_1_s, full_history_2=full_obj_2_s)
        self.exec_time = time.time() - t
        if verbose:
            self.log(
                f"Took {time_formatter(self.exec_time)} to optimise {self.designs_group_id} "
                f"-> improvement QoR is {((2 - objs.sum(-1).min()) / 2) * 100:.2f}%")
        return res

    def process_results(self, res: MultiRandomRes) -> pd.DataFrame:
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

        pd_res['qor'] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values('qor')

    def get_samples(self) -> np.ndarray:
        """ Return either an array of initial population obtained with latin hypercube sampling """
        samples = sample(self.sampler, n_samples=self.n_trials, n_var=self.seq_length, xl=0,
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

        return samples

    def save_results(self, res: MultiRandomRes) -> None:
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
        log(msg, header=self.method_id, end=end)

    @staticmethod
    def get_history_values(res: MultiRandomRes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)
