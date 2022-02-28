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
from abc import ABC, abstractmethod
from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common import BaseRLModel, ActorCriticRLModel
from stable_baselines.common.policies import LstmPolicy, FeedForwardPolicy
from stable_baselines.deepq import LnMlpPolicy, MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as ppoMlpPolicy
from typing import Type, List, Dict, Union

from DRiLLS.drills.fpga_session import FPGASessionEnv
from DRiLLS.drills.models.agent import Agent
from DRiLLS.utils import _filter_kwargs
from utils.utils_save import str_list, get_storage_root


class AgentGym(Agent, ABC):

    @abstractmethod
    def __init__(self, env: FPGASessionEnv, rl_algo: Type[BaseRLModel], seed: int, **kwargs):
        self.model = rl_algo(env=env, **kwargs)
        self.design = env.fpgasess.design_name
        self.max_iterations = env.fpgasess.max_iterations
        self.action_space_id = env.fpgasess.action_space_id
        self.lut_inputs = env.fpgasess.lut_inputs
        self.objective = env.fpgasess.objective
        self.ref_abc_seq = env.fpgasess.ref_abc_seq
        self.seed = seed

        os.makedirs(os.path.dirname(self.model_path()), exist_ok=True)

    @staticmethod
    @abstractmethod
    def model_static_id(**kwargs) -> str:
        pass

    def train(self, n_episodes: int):
        self.model.learn(total_timesteps=n_episodes * (self.max_iterations - 1))

    def model_path(self) -> str:
        return self.get_model_path(
            design=self.design,
            learner_id=self.model_id,
            action_space_id=self.action_space_id,
            lut_inputs=self.lut_inputs,
            objective=self.objective,
            ref_abc_seq=self.ref_abc_seq,
            seed=self.seed
        )

    @staticmethod
    def get_model_path(design: str, learner_id: str, action_space_id: str, lut_inputs: int,
                       objective: str, ref_abc_seq: str, seed: int):
        model_path_id = f"act-space-{action_space_id}_lut-{lut_inputs}_obj-{objective}_ref-seq-{ref_abc_seq}"
        return os.path.join(
            get_storage_root(),
            'model',
            learner_id,
            f"{design}",
            model_path_id,
            str(seed),
            'checkpoints')

    def save(self):
        self.model.save(self.model_path())


class AgentPPO(AgentGym):

    def __init__(self, env, **kwargs):
        super().__init__(env, PPO, policy=ppoMlpPolicy, verbose=2, **kwargs)

    def train(self, n_episodes: int):
        self.model.learn(total_timesteps=n_episodes * (self.max_iterations - 1),
                         reset_num_timesteps=False)

    @property
    def model_id(self) -> str:
        return 'ppo'

    @staticmethod
    def model_static_id() -> str:
        return 'ppo'


class AgentOnPolicy(AgentGym, ABC):

    def __init__(self, env, seed: int, rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int], layer_norm: bool,
                 feature_extraction: str, ent_coef: float, rl_algo: Type[ActorCriticRLModel], learning_rate: float,
                 n_steps: int):
        assert feature_extraction in ['mlp', 'cnn']
        self.feature_extraction = feature_extraction
        self.net_arch: List[Union[str, int, Dict[str, List[int]]]] = ['lstm' if rec else 0, {}]
        if pi_arch is None:
            pi_arch = [20, 20]
        if vf_arch is None:
            vf_arch = [10]
        self.net_arch[-1]['pi'] = pi_arch
        self.net_arch[-1]['vf'] = vf_arch
        self.pi_arch = pi_arch
        self.vf_arch = vf_arch
        self.layer_norm = layer_norm
        self.rec = rec
        self.feature_extraction = feature_extraction
        self.n_lstm = n_lstm
        self.learning_rate = learning_rate
        self.policy = LstmPolicy if rec else FeedForwardPolicy
        self.ent_coef = ent_coef
        policy_kwargs = dict(net_arch=self.net_arch, layer_norm=self.layer_norm,
                             feature_extraction=self.feature_extraction,
                             n_lstm=n_lstm
                             )
        self.policy_kwargs = _filter_kwargs(self.policy, **policy_kwargs)
        kwargs = dict(policy_kwargs=self.policy_kwargs, n_steps=n_steps, policy=self.policy, ent_coef=self.ent_coef,
                      learning_rate=self.learning_rate, nminibatches=1)
        super().__init__(env, rl_algo, seed, **_filter_kwargs(rl_algo, **kwargs))

    @staticmethod
    def get_model_id_ext(rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int], layer_norm: bool,
                         feature_extraction: str, ent_coef: float, learning_rate: float) -> str:
        assert feature_extraction in ['cnn', 'mlp']
        model_id = ''
        if rec:
            model_id += f'_lstm-{n_lstm}'
        model_id += f'_pi-{str_list(pi_arch)}'
        model_id += f'_vf-{str_list(vf_arch)}'
        if layer_norm:
            model_id += f'_ln'
        if feature_extraction == 'cnn':
            model_id += f'_cnn'
        if ent_coef != .01:
            model_id += f'_ent-{float(ent_coef)}'
        model_id += f'_lr-{learning_rate}'
        return model_id


class AgentA2C(AgentOnPolicy):
    def __init__(self, env: FPGASessionEnv, seed: int, rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int],
                 layer_norm: bool,
                 feature_extraction: str, ent_coef: float, learning_rate: float):
        super().__init__(
            env=env,
            seed=seed,
            rec=rec,
            n_lstm=n_lstm,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            layer_norm=layer_norm,
            feature_extraction=feature_extraction,
            ent_coef=ent_coef,
            rl_algo=A2C,
            learning_rate=learning_rate,
            n_steps=5
        )

    def train(self, n_episodes: int):
        self.model.learn(total_timesteps=n_episodes * (self.max_iterations - 1), reset_num_timesteps=False)

    @property
    def model_id(self) -> str:
        return self.model_static_id(
            rec=self.rec,
            n_lstm=self.n_lstm,
            pi_arch=self.pi_arch,
            vf_arch=self.vf_arch,
            layer_norm=self.layer_norm,
            feature_extraction=self.feature_extraction,
            ent_coef=self.ent_coef,
            learning_rate=self.learning_rate
        )

    @staticmethod
    def model_static_id(rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int], layer_norm: bool,
                        feature_extraction: str, ent_coef: float, learning_rate: float) -> str:
        return 'a2c' + AgentOnPolicy.get_model_id_ext(
            rec=rec,
            n_lstm=n_lstm,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            layer_norm=layer_norm,
            feature_extraction=feature_extraction,
            ent_coef=ent_coef,
            learning_rate=learning_rate
        )


class AgentPPO2(AgentOnPolicy):
    model: PPO2

    def __init__(self, env: FPGASessionEnv, rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int],
                 layer_norm: bool,
                 feature_extraction: str, ent_coef: float, learning_rate: float, seed: int):
        super().__init__(
            env=env,
            rec=rec,
            seed=seed,
            n_lstm=n_lstm,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            layer_norm=layer_norm,
            feature_extraction=feature_extraction,
            ent_coef=ent_coef,
            rl_algo=PPO2,
            learning_rate=learning_rate,
            n_steps=env.fpgasess.max_iterations - 1
        )

    def train(self, n_episodes: int):
        self.model.learn(total_timesteps=n_episodes * (self.max_iterations - 1),
                         reset_num_timesteps=False)

    @property
    def model_id(self) -> str:
        return self.model_static_id(
            rec=self.rec,
            n_lstm=self.n_lstm,
            pi_arch=self.pi_arch,
            vf_arch=self.vf_arch,
            layer_norm=self.layer_norm,
            feature_extraction=self.feature_extraction,
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef
        )

    @staticmethod
    def model_static_id(rec: bool, n_lstm: int, pi_arch: List[int], vf_arch: List[int], layer_norm: bool,
                        feature_extraction: str, ent_coef: float, learning_rate: float) -> str:
        return 'ppo2' + AgentOnPolicy.get_model_id_ext(
            rec=rec,
            n_lstm=n_lstm,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            layer_norm=layer_norm,
            feature_extraction=feature_extraction,
            ent_coef=ent_coef,
            learning_rate=learning_rate
        )


class AgentDQN(AgentGym):

    def __init__(self, env: FPGASessionEnv, double_q: bool, dueling: bool, layer_norm: bool, learning_rate: float,
                 seed: int):
        self.dueling = dueling
        self.layer_norm = layer_norm
        self.learning_rate = learning_rate
        self.double_q = double_q
        if self.layer_norm:
            policy = LnMlpPolicy
        else:
            policy = MlpPolicy
        policy_kw = dict(dueling=self.dueling)
        kwargs = dict(policy=policy, learning_starts=100, learning_rate=self.learning_rate, double_q=self.double_q,
                      policy_kwargs=policy_kw)
        super().__init__(env, DQN, seed=seed, verbose=1, **kwargs)

    @property
    def model_id(self) -> str:
        return self.model_static_id(
            layer_norm=self.layer_norm,
            double_q=self.double_q,
            dueling=self.dueling,
            learning_rate=self.learning_rate
        )

    def train(self, n_episodes: int):
        self.model.learn(total_timesteps=n_episodes * (self.max_iterations - 1),
                         reset_num_timesteps=False)

    @staticmethod
    def model_static_id(layer_norm: bool, double_q: bool, dueling: bool, learning_rate: float) -> str:
        model_id = 'dqn'
        if layer_norm:
            model_id += f'_ln'
        if dueling:
            model_id += f'_duel'
        if double_q:
            model_id += f'_doubq'
        model_id += f'_lr-{learning_rate}'
        return model_id
