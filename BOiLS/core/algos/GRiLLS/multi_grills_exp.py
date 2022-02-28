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
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from resources.abcRL import PiApprox, FcModelGraph, FcModel, BaselineVApprox
from core.algos.GRiLLS.grills_env import EnvGraph
from core.algos.GRiLLS.grills_reinforce import Reinforce
from core.algos.GRiLLS.utils import METHOD_ID, META_METHOD_ID
from core.algos.common_exp import MultiEADExp
from core.algos.utils import Res, is_pareto_efficient, get_history_values_from_res
from utils.utils_misc import time_formatter, log
from utils.utils_save import save_w_pickle


class MultiGRiLLSRes(Res):
    pass


def obj_both(ratio_1, ratio_2):
    return ratio_1 + ratio_2


def obj_level(ratio_1, ratio_2):
    return ratio_2


def obj_lut(ratio_1, ratio_2):
    return ratio_1


def obj_min_improvements(ratio_1, ratio_2):
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


class MultiGRiLLSExp(MultiEADExp):
    color = 'brown'

    method_id: str = METHOD_ID
    meta_method_id: str = META_METHOD_ID

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, lut_inputs: int, action_space_id: str,
                 library_file: str,
                 abc_binary: str, ref_abc_seq: Optional[str], objective: str, use_yosys: bool,
                 n_episodes: int, seed: int, alpha_pi: float, alpha_v: float, gamma: float = .9):
        """
        Args:
            designs_group_id: id of the group of designs
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            n_episodes: number of RL episodes to run
            objective: quantity to optimize, either lut, level, both or min_improvements
            seed: seed for reproducibility
            alpha_pi: policy learning rate
            alpha_v: value function learning rate
            gamma: decaying rate
        """
        super().__init__(designs_group_id=designs_group_id,
                         seq_length=seq_length,
                         mapping=mapping,
                         action_space_id=action_space_id,
                         library_file=library_file,
                         abc_binary=abc_binary,
                         lut_inputs=lut_inputs,
                         ref_abc_seq=ref_abc_seq,
                         use_yosys=use_yosys)

        assert len(self.design_files) == 1, "Multi-design not yet supported"

        self.n_episodes = n_episodes
        self.seed = seed
        self.alpha_pi = alpha_pi
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.exec_time_eps: List[float] = []
        self.ref_time = time.time()

        self.piApprox: Optional[PiApprox] = None
        self.vbaseline: Optional[BaselineVApprox] = None

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

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiGRiLLSExp, self).get_config()
        config['n_episodes'] = self.n_episodes
        config['seed'] = self.seed
        config['alpha_pi'] = self.alpha_pi
        config['alpha_v'] = self.alpha_v
        config['gamma'] = self.gamma
        config['objective'] = self.objective,
        return config

    def save_results(self, res: MultiGRiLLSRes) -> None:
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

        # save vbaseline
        self.vbaseline.save(os.path.join(save_path, 'vbaseline.pt'))
        # save piApprox
        self.piApprox.save(os.path.join(save_path, 'piApprox.pt'))

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

    @staticmethod
    def get_exp_id(objective: str, use_yosys: bool, n_episodes: int, alpha_pi: float, alpha_v: float,
                   gamma: float) -> str:
        exp_id = MultiGRiLLSExp.method_id
        exp_id += f"_ep-{n_episodes}"
        exp_id += f"_obj-{objective}"
        exp_id += f"_alpha-pi-{alpha_pi:g}"
        exp_id += f"_alpha-v-{alpha_v:g}"
        exp_id += f"_gamma-{gamma:g}"
        if use_yosys:
            exp_id += '_yosys'
        return exp_id

    def exp_id(self) -> str:
        return self.get_exp_id(
            objective=self.objective,
            use_yosys=self.use_yosys,
            n_episodes=self.n_episodes,
            alpha_pi=self.alpha_pi,
            alpha_v=self.alpha_v,
            gamma=self.gamma,
        )

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
        return os.path.join(MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiGRiLLSExp.meta_method_id,
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

    def run(self, verbose: bool = True):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        t = time.time()

        history_x = []
        obj_1_s = []
        obj_2_s = []

        assert len(self.design_files) == 1

        env: EnvGraph = EnvGraph(design_file=self.design_files[0], library_file=self.library_file,
                                 abc_binary=self.abc_binary, objective=self.objective, use_yosys=self.use_yosys,
                                 mapping=self.mapping, lut_inputs=self.lut_inputs, seq_length=self.seq_length,
                                 ref_obj_1=self.refs_1[0], ref_obj_2=self.refs_2[0], action_space=self.action_space,
                                 playground_path=self.exp_path())

        self.piApprox = PiApprox(env.dim_state(), env.num_actions, alpha=self.alpha_pi, network=FcModelGraph)
        self.vbaseline = BaselineVApprox(env.dim_state(), alpha=self.alpha_v, network=FcModel)
        reinforce = Reinforce(env, gamma=self.gamma, pi=self.piApprox, baseline=self.vbaseline)

        for episode in range(self.n_episodes):
            t_ep = time.time()
            reinforce.episode(phaseTrain=True)
            self.exec_time_eps.append(time.time() - t_ep)

            history_x.append(env.episode_hist.action_ind_seq)
            obj_1_s.append(env.episode_hist.objs_1[-1] / self.refs_1[0])
            obj_2_s.append(env.episode_hist.objs_2[-1] / self.refs_2[0])

            if env.episode % 10 == 0:
                self.log(
                    f'Optim {self.design_names[0]} with {self.action_space_id} act.:'
                    f' Episode {env.episode} / {self.n_episodes}'
                    f' -> took {time_formatter(time.time() - t)}')
                objs = np.stack([obj_1_s, obj_2_s]).T
                save_path = self.exp_path()
                self.log(f'{self.exp_id()} -> Save checkpoint to {save_path}...')
                os.makedirs(save_path, exist_ok=True)
                save_w_pickle({"objs": objs, 'history_x': np.array(history_x)},
                              os.path.join(save_path, "checkpoint.pkl"))

        objs = np.stack([obj_1_s, obj_2_s]).T
        assert objs.shape == (self.n_episodes, 2), objs.shape
        mask_pareto = is_pareto_efficient(objs)
        history_x = np.array(history_x)
        history_f = objs.copy()
        samples = history_x[mask_pareto]
        objs = objs[mask_pareto]
        res = MultiGRiLLSRes(X=samples, F=objs, history_x=history_x, history_f=history_f)

        self.exec_time = time.time() - t
        if verbose:
            aux1 = objs[:, 0]
            aux2 = objs[:, 1]
            self.log(
                f"Took {time_formatter(self.exec_time)} to optimise {self.design_names[0]} "
                f"-> improvement QoR is {((2 - (aux1 + aux2).min()) / 2) * 100:.2f}%")
        return res

    def process_results(self, res: MultiGRiLLSRes) -> pd.DataFrame:
        seq_id = []
        obj_1 = []
        ratio_1 = []
        obj_2 = []
        ratio_2 = []
        for seq_ind, func_value in zip(res.X, res.F):
            seq_id.append(' | '.join([self.action_space[ind].act_id for ind in seq_ind]))
            ratio_1.append(func_value[0])
            ratio_2.append(func_value[1])
            obj_1.append(ratio_1[-1] * self.refs_1[0])
            obj_2.append(ratio_2[-1] * self.refs_2[0])
        pd_res = pd.DataFrame()
        pd_res['seq_id'] = seq_id

        pd_res[self.obj1_id] = obj_1
        pd_res[self.obj2_id] = obj_2

        pd_res['ratio ' + self.obj1_id] = ratio_1
        pd_res['ratio ' + self.obj2_id] = ratio_2

        pd_res['qor'] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values('qor')

    def log(self, msg: str, end=None) -> None:
        log(msg, header=self.method_id, end=end)

    @staticmethod
    def get_history_values(res: MultiGRiLLSRes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)
