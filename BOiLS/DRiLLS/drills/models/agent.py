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

import abc
from abc import abstractmethod

from DRiLLS.baseline.greedy.greedy_session import GreedySclSession
from DRiLLS.drills.scl_session import SCLSessionEnv
from DRiLLS.utils import get_model_path


class Agent(abc.ABC):

    @abstractmethod
    def train(self, n_episodes: int):
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def model_path(self) -> str:
        pass


class AgentRandom(Agent):

    def __init__(self, design: str, env: SCLSessionEnv):
        self.design = design
        self.env = env

    def train(self, n_episodes: int):
        for episode in range(n_episodes):
            self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                _, _, done, _ = self.env.step(action)

    @property
    def model_id(self) -> str:
        return self.model_static_id()

    @staticmethod
    def model_static_id() -> str:
        return 'random-search'

    def model_path(self) -> str:
        return get_model_path(
            design=self.design,
            learner_id=self.model_id
        )
