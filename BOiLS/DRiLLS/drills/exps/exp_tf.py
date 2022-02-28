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
from copy import deepcopy
from typing import List

import yaml
import time

import numpy as np

from DRiLLS.drills.exps.exp import RLExp
from DRiLLS.drills.model import A2C
from utils.utils_misc import log
from utils.utils_save import ROOT_PROJECT

# DEPRECATED
class ExpTF(RLExp):
    exec_time: float

    def __init__(self, params_file: str, design_file: str, episodes: int, max_iteration, mapping: str,
                 load_model_from: str = None):
        super().__init__(
            params_file=params_file,
            design_file=design_file,
            load_model_from=load_model_from,
            episodes=episodes,
            max_iteration=max_iteration,
            mapping=mapping)

        self.learner: A2C = A2C(self.options, design=self.design, design_file=self.design_file,
                                max_iteration=self.max_iteration,
                                load_model_from=self.load_model_from, fpga_mapping=self.fpga_mapping)

        self.playground_dir = self.learner.playground_dir

    def train(self):
        """ Train an agent """
        t = time.time()
        self.learner.train(n_episodes=self.episodes)
        self.exec_time = time.time() - t
        np.save(os.path.join(self.playground_dir, 'exec_time.npy'), self.exec_time)
        # mean_reward = np.mean(all_rewards[-100:])

    def optimize(self, params_file: str, design_files: List[str], max_iterations: int, overwrite: bool):
        raise ValueError()
    #     print(f"Load from {self.get_model_path()}")
    #     load_from_model = self.get_model_path()
    #     constraints_config = yaml.load(open(os.path.join(ROOT_PROJECT, 'data', 'constraints.yaml')),
    #                                    Loader=yaml.FullLoader)
    #     for design_file in design_files:
    #         options = deepcopy(self.options)
    #
    #         design = design_file.split('.')[0]
    #
    #         options['mapping']['clock_period'] = constraints_config['delay_constraints'][design]
    #         print(
    #             f"No delay constraint provided: take {options['mapping']['clock_period']} for {design}")
    #
    #         design_file = os.path.join(ROOT_PROJECT, 'data', 'epfl-benchmark', 'arithmetic', design_file)
    #         playground_dir = os.path.join(self.playground_dir, 'test', design)
    #         if self.already_trained(playground_dir=playground_dir) and not overwrite:
    #             print(f"Already tested {self.learner.model_id} trained on {self.design} on {design}: {playground_dir}")
    #             continue
    #         learner: A2C = A2C(options, design=design, design_file=design_file, max_iteration=max_iterations,
    #                            load_model_from=load_from_model, fpga_mapping=self.fpga_mapping)
    #
    #         area, delay, met = learner.run_episode()
    #         log(f"Area: {area:g} | Delay: {delay} | Constrained {'not ' if not met else ''}met")
    #
    def get_model_path(self) -> str:
        """ Path where model checkpoints will be stored """
        return self.learner.model_path()
