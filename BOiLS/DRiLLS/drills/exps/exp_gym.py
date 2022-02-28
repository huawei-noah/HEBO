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
import torch
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from typing import Type, List

from DRiLLS.drills.exps.exp import RLExp
from DRiLLS.drills.fpga_session import FPGASession, FPGASessionEnv
from DRiLLS.drills.models.gym_agents import AgentGym, AgentPPO, AgentDQN
from DRiLLS.utils import _filter_kwargs
from core.design_groups import get_designs_path
from utils.utils_save import save_w_pickle


class ExpGym(RLExp):

    def __init__(self, design_id: str, episodes: int, max_iteration,
                 agent_class: Type[AgentGym], mapping: str, lut_inputs: int,
                 ref_abc_seq: str,
                 objective: str, action_space_id: str, seed: int,
                 load_model_from: str = None, **agent_kwargs):
        super().__init__(
            design_id=design_id,
            load_model_from=load_model_from,
            episodes=episodes,
            max_iteration=max_iteration,
            mapping=mapping, lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq, objective=objective, action_space_id=action_space_id, seed=seed)

        self.playground_dir = self.get_playground_dir(
            design=self.design_id,
            exp_id=self.exp_id,
            learner_id=agent_class.model_static_id(**_filter_kwargs(agent_class.model_static_id, **agent_kwargs)),
            seed=self.seed
        )

        self.game = FPGASession(
            design_name=self.design_id,
            design_file=self.design_file,
            playground_dir=self.playground_dir,
            mapping=self.mapping, lut_inputs=self.lut_inputs, abc_binary=self.abc_binary, ref_abc_seq=self.ref_abc_seq,
            objective=self.objective,
            action_space_id=self.action_space_id,
            max_iterations=self.max_iteration
        )

        torch.manual_seed(seed=self.seed)
        self.env: FPGASessionEnv = FPGASessionEnv(
            self.game
        )
        self.agent_class = agent_class
        self.learner: AgentGym = agent_class(env=self.env, seed=seed, **agent_kwargs)
        assert self.playground_dir == self.env.fpgasess.playground_dir, (self.playground_dir,
                                                                         self.env.fpgasess.playground_dir)

    def train(self):
        self.learner.train(n_episodes=self.episodes)
        save_w_pickle(self.env.fpgasess.hist, self.playground_dir, 'hist.pkl')
        self.env.fpgasess.clean()

    def optimize(self, design_ids: List[str], max_iterations: int, overwrite: bool):
        print(f"Load from {self.get_model_path()}")
        self.learner.model.load(load_path=self.get_model_path())

        for design_id in design_ids:
            design_file = get_designs_path(self.design_id)[0]
            playground_dir = os.path.join(self.playground_dir, 'test', design_id)
            game = FPGASession(
                design_name=design_id,
                design_file=design_file,
                playground_dir=playground_dir,
                action_space_id=self.action_space_id,
                mapping=self.mapping,
                lut_inputs=self.lut_inputs,
                abc_binary=self.abc_binary,
                ref_abc_seq=self.ref_abc_seq,
                objective=self.objective,
                max_iterations=max_iterations
            )

            if hasattr(self.learner, 'rec') and self.learner.rec:
                env = DummyVecEnv([lambda: FPGASessionEnv(
                    game
                )])
            else:
                env = FPGASessionEnv(
                    game
                )
            if self.already_trained(playground_dir=playground_dir) and not overwrite:
                print(f"Already tested {self.learner.model_id} trained on {self.design_id} on {design_id}:"
                      f" {playground_dir}")
                continue
            evaluate_policy(self.learner.model, env=env, n_eval_episodes=1)


class ExpPPO(ExpGym):

    def __init__(self, design_id: str, episodes: int, max_iteration: int,
                 mapping: str, lut_inputs: int, ref_abc_seq: str, objective: str,
                 action_space_id: str, seed: int,
                 load_model_from: str = None, **agent_kwargs):
        super().__init__(
            design_id=design_id,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq,
            objective=objective,
            action_space_id=action_space_id,
            seed=seed,
            load_model_from=load_model_from,
            episodes=episodes,
            max_iteration=max_iteration,
            mapping=mapping,
            agent_class=AgentPPO,
            n_steps=max_iteration,
            **agent_kwargs
        )


class ExpDQN(ExpGym):

    def __init__(self, design_id: str, episodes: int, max_iteration,
                 mapping: str, lut_inputs: int, ref_abc_seq: str, objective: str,
                 action_space_id: str, seed: int,
                 double_q: bool, dueling: bool, layer_norm: bool, learning_rate: float,
                 load_model_from: str = None):
        super().__init__(
            design_id=design_id,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq,
            objective=objective,
            action_space_id=action_space_id,
            seed=seed,
            load_model_from=load_model_from,
            episodes=episodes,
            max_iteration=max_iteration,
            mapping=mapping,
            agent_class=AgentDQN,
            double_q=double_q,
            dueling=dueling,
            layer_norm=layer_norm,
            learning_rate=learning_rate,
        )


class ExpOnPolicy(ExpGym):

    def __init__(self, agent_class, design_id: str, episodes: int, max_iteration,
                 rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int],
                 layer_norm: bool,
                 feature_extraction: str, ent_coef: float, learning_rate: float,
                 mapping: str, lut_inputs: int, ref_abc_seq: str, objective: str,
                 action_space_id: str, seed: int,
                 load_model_from: str = None):
        super().__init__(
            design_id=design_id,
            load_model_from=load_model_from,
            episodes=episodes,
            max_iteration=max_iteration,
            mapping=mapping,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq,
            objective=objective,
            action_space_id=action_space_id,
            seed=seed,
            agent_class=agent_class,
            rec=rec,
            n_lstm=n_lstm,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            layer_norm=layer_norm,
            feature_extraction=feature_extraction,
            ent_coef=ent_coef,
            learning_rate=learning_rate
        )
