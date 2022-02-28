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

import numpy  as np
import os
import pandas as pd
import torch
import shutil
import time
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from joblib import Parallel, delayed
from subprocess import CalledProcessError
from typing import Dict, Any, Optional, Tuple, List

from core.action_space import Action
from core.algos.common_exp import MultiEADExp, Checkpoint
from core.algos.utils import is_pareto_efficient, Res, get_history_values_from_res, get_design_name
from core.sessions.utils import get_design_prop
from utils.utils_misc import log, time_formatter
from utils.utils_save import save_w_pickle, load_w_pickle


class MultiHeboRes(Res):
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


def obj_both(ratio_1, ratio_2):
    return ratio_1 + ratio_2


def obj_level(ratio_1, ratio_2):
    return ratio_2


def obj_lut(ratio_1, ratio_2):
    return ratio_1


def obj_min_improvements(ratio_1, ratio_2):
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


class MultiHeboExp(MultiEADExp):
    """ Class associated to HEBO to solve QoR minimization """

    color = 'purple'

    method_id: str = 'HEBO'
    meta_method_id: str = 'BO'
    n_suggestions = 1

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int, n_initial: int,
                 lut_inputs: int,
                 use_yosys: bool,
                 ref_abc_seq: Optional[str], objective: str, overwrite: bool,
                 n_parallel: int = 1):
        """
        Args:
            designs_group_id: id of the group of designs
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            use_yosys: whether to use yosys-abc or abc_py
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            seed: reproducibility seed
            n_initial: number of initial points to test before building first surrogate model
            objective: quantity to optimize, either lut, level, both or min_improvements
       """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, abc_binary=abc_binary,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq, use_yosys=use_yosys)
        self.seed = seed

        self.n_initial = n_initial
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
        config = super(MultiHeboExp, self).get_config()
        config['seed'] = self.seed
        config['n_initial'] = self.n_initial
        config['objective'] = self.objective,
        return config

    @staticmethod
    def get_exp_id(n_initial: int, objective: str, use_yosys: bool) -> str:
        exp_id = MultiHeboExp.method_id
        exp_id += f"_init-{n_initial}"
        exp_id += f"_obj-{objective}"
        if use_yosys:
            exp_id += '_yosys'
        return exp_id

    def exp_id(self) -> str:
        return self.get_exp_id(
            n_initial=self.n_initial,
            objective=self.objective,
            use_yosys=self.use_yosys
        )

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
        return os.path.join(MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiHeboExp.meta_method_id,
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

    def run(self, n_total_evals: int, verbose: bool = False):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        space = DesignSpace().parse([
            {
                'name': f'act_{act_ind}',
                'type': 'cat',
                'categories': list(map(str, np.arange(self.action_space_length)))
            }
            for act_ind in range(self.seq_length)
        ])

        ckpt_path = os.path.join(self.exp_path(), 'ckpt.pkl')
        opt = HEBO(space, rand_sample=self.n_initial)
        if os.path.exists(ckpt_path):
            ckpt: Checkpoint = load_w_pickle(ckpt_path)
            self.samples_X = ckpt.samples[:n_total_evals]
            self.full_obj_1_s = list(ckpt.full_objs_1[:len(self.samples_X)])
            self.full_obj_2_s = list(ckpt.full_objs_2[:len(self.samples_X)])
            self.n_evals = len(self.samples_X)
            aux = opt.quasi_sample(self.samples_X.shape[0])
            aux[:] = self.samples_X.astype(str)
            opt.observe(aux, np.array([np.mean([self.objective_function(o1, o2) for (o1, o2) in
                        zip(self.full_obj_1_s[i], self.full_obj_2_s[i])]) for i in range(len(self.samples_X))]))
            opt.sobol.reset()
            opt.sobol.fast_forward(min(self.n_initial, len(ckpt.samples)))
            self.samples_X = list(self.samples_X)

        for i in range(len(self.samples_X), n_total_evals, self.n_suggestions):
            x_next: pd.DataFrame = opt.suggest(n_suggestions=self.n_suggestions)
            y_next: np.ndarray = self.evaluate(x_next.values, iter=i)
            opt.observe(x_next, y_next.reshape(len(x_next), -1))

        return self.build_res(verbose=verbose)

    def evaluate(self, x: np.ndarray, iter: int) -> np.ndarray:
        """

        Args:
            x: new point to evaluate
        """
        # self.log(f"{self.n_evals:3d}. Evaluate sequence {x} on design {self.design_name}: ", end="")
        X = x.astype(int).reshape(self.n_suggestions, self.seq_length)

        ind_prod = list(
            itertools.product(range(len(self.design_files)), range(0, min(self.n_parallel, len(X)))))

        objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
            delayed(multi_hebo_exp_get_obj)(sequence=X[ind_sample],
                                            design_file=self.design_files[ind_design],
                                  ref_1=self.refs_1[ind_design],
                                  ref_2=self.refs_2[ind_design],
                                            action_space=self.action_space,
                                            playground=self.playground,
                                            mapping=self.mapping,
                                            library_file=self.library_file,
                                            abc_binary=self.abc_binary,
                                            lut_inputs=self.lut_inputs,
                                            n_evals=self.n_evals + ind_sample,
                                            use_yosys=self.use_yosys)
            for ind_design, ind_sample in ind_prod
        )
        full_obj_1_s = [np.zeros(len(self.design_files)) for _ in range(len(X))]
        full_obj_2_s = [np.zeros(len(self.design_files)) for _ in range(len(X))]
        valids = [np.zeros(len(self.design_files)) for _ in range(len(X))]
        self.n_evals += len(X)
        for j in range(len(ind_prod)):
            ind_design, ind_sample = ind_prod[j]
            full_obj_1_s[ind_sample][ind_design] = objs[j][0]
            full_obj_2_s[ind_sample][ind_design] = objs[j][1]
            valids[ind_sample][ind_design] = int(objs[j][2])

        # objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
        #     delayed(multi_hebo_exp_get_obj)(
        #         sequence=X, design_file=self.design_files[k], ref_1=self.refs_1[k],
        #         ref_2=self.refs_2[k],
        #         action_space=self.action_space,
        #         playground=self.playground,
        #         mapping=self.mapping,
        #         library_file=self.library_file,
        #         abc_binary=self.abc_binary,
        #         lut_inputs=self.lut_inputs,
        #         n_evals=self.n_evals,
        #         use_yosys=self.use_yosys
        #     ) for k in range(len(self.design_files)))
        self.full_obj_1_s.extend(full_obj_1_s)
        self.full_obj_2_s.extend(full_obj_2_s)
        self.valids.extend(valids)
        self.samples_X.extend(X)

        save_w_pickle(
            Checkpoint(samples=np.array(self.samples_X), full_objs_1=np.array(self.full_obj_1_s),
                       full_objs_2=np.array(self.full_obj_2_s)),
            path=self.exp_path(),
            filename='ckpt.pkl'
        )

        return np.array([np.mean([self.objective_function(o1, o2) for (o1, o2) in
                        zip(self.full_obj_1_s[-i], self.full_obj_2_s[-i])]) for i in range(len(X))])


    def build_res(self, verbose: bool = False) -> MultiHeboRes:
        objs = np.stack([np.mean(self.full_obj_1_s, 1), np.mean(self.full_obj_2_s, 1)]).T
        assert objs.shape == (self.n_evals, 2), (objs.shape, self.n_evals)
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
        history_x = np.array(self.samples_X).copy()
        history_f = objs.copy()
        samples = history_x[mask]
        objs = objs[mask]
        res = MultiHeboRes(X=samples, F=objs, history_x=history_x, history_f=history_f, mask=mask,
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

    def process_results(self, res: MultiHeboRes) -> pd.DataFrame:
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

    def save_results(self, res: MultiHeboRes) -> None:
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

    def exists(self, n_total_evals: Optional[int] = None) -> bool:
        """ Check if experiment already exists """
        save_path = self.exp_path()
        paths_to_check = [
            os.path.join(save_path, 'res.csv'),
            os.path.join(save_path, 'exec_time.npy'),
            os.path.join(save_path, 'config.pkl'),
            os.path.join(save_path, 'res.pkl')
        ]
        if not np.all(list(map(lambda p: os.path.exists(p), paths_to_check))):
            return False
        # check that enough sequences have been tested
        if n_total_evals is None:
            return True
        return load_w_pickle(os.path.join(save_path, 'res.pkl')).history_x.shape[0] >= n_total_evals

    @staticmethod
    def get_history_values(res: MultiHeboRes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)


def multi_hebo_exp_get_obj(sequence: List[int], design_file: str, ref_1: float, ref_2: float,
                           action_space: List[Action],
                           playground: str, mapping: str, library_file: str, abc_binary: str, use_yosys: bool,
                           lut_inputs: int, n_evals: int) \
        -> Tuple[float, float, bool]:
    """ Return either area and delay or lut and levels """
    sequence = [(action_space[ind].act_id if not use_yosys else action_space[ind].act_str) for ind in sequence]
    sequence_id = ' '.join(map(str, sequence))
    save_file = os.path.join(playground, get_design_name(design_file), 'seq_to_func_dic.pkl')
    seq_to_func_dic: Dict[str, Tuple[float, float, bool]] = {}
    if not os.path.exists(save_file):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        save_w_pickle(seq_to_func_dic, os.path.dirname(save_file), os.path.basename(save_file))
    seq_to_func_dic = load_w_pickle(os.path.dirname(save_file), os.path.basename(save_file))

    if sequence_id not in seq_to_func_dic:
        valid = True
        try:
            log(f"{n_evals}. Evaluate {sequence_id} for {get_design_name(design_file)} ", header="HEBO ")
            obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=design_file, mapping=mapping,
                                                       library_file=library_file, abc_binary=abc_binary,
                                                       use_yosys=use_yosys,
                                                       lut_inputs=lut_inputs, compute_init_stats=False)
        except CalledProcessError as e:
            if e.args[0] == -6:
                log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ", header="HEBO ")
                obj_1 = ref_1
                obj_2 = ref_2
                valid = False
            else:
                raise e
        seq_to_func_dic = load_w_pickle(os.path.dirname(save_file), os.path.basename(save_file))
        seq_to_func_dic[sequence_id] = obj_1 / ref_1, obj_2 / ref_2, valid
        save_w_pickle(seq_to_func_dic, os.path.dirname(save_file), os.path.basename(save_file))
    else:
        log(f"{n_evals}. Already computed {sequence_id} for {get_design_name(design_file)}...", header="HEBO ")
    return seq_to_func_dic[sequence_id]
