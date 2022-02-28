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
import time
from joblib import Parallel, delayed
from subprocess import CalledProcessError
from typing import List, Dict

from DRiLLS.drills.exps.exp import ExpABC
from DRiLLS.drills.fpga_session import FPGASession
from core.sessions.utils import get_design_prop
from utils.utils_misc import log
from utils.utils_save import get_storage_root, save_w_pickle


class ExpGreedy(ExpABC):
    method_id: str = 'Greedy'

    def __init__(self, design_id: str, lut_inputs: int, max_iteration,
                 ref_abc_seq: str, mapping: str, objective: str,
                 action_space_id: str, seed: int, n_parallel: int,
                 ):
        """

        Args:
            design_id:
            lut_inputs:
            max_iteration:
            ref_abc_seq:
            mapping:
            objective:
            action_space_id:
            seed:
            n_parallel: parallelise on the actions in action space (test them in parallel to gain some time)
        """
        super().__init__(
            design_id=design_id,
            max_iteration=max_iteration,
            mapping=mapping,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq,
            objective=objective,
            action_space_id=action_space_id,
            seed=seed
        )

        np.random.seed(seed)

        self.n_parallel = n_parallel
        self.timings: List[Dict[str, float]] = []
        self.playground_dir = self.get_playground_dir(
            design=self.design_id,
            exp_id=self.exp_id,
            seed=self.seed
        )

    @staticmethod
    def get_playground_dir(
            design,
            exp_id,
            seed,
    ) -> str:
        playground_dir = os.path.join(get_storage_root(), ExpGreedy.method_id,
                                      'playground', design, exp_id, str(seed))
        return playground_dir

    @property
    def exp_id(self) -> str:
        return self.get_exp_id(
            action_space_id=self.action_space_id,
            lut_inputs=self.lut_inputs,
            objective=self.objective,
            ref_abc_seq=self.ref_abc_seq,
            seq_length=self.max_iteration,
        )

    @staticmethod
    def get_exp_id(action_space_id: str, lut_inputs: int, objective: str, ref_abc_seq: str, seq_length: int,
                   ):
        return ExpABC.get_exp_id_basis(
            action_space_id=action_space_id,
            lut_inputs=lut_inputs,
            objective=objective,
            ref_abc_seq=ref_abc_seq,
            seq_length=seq_length
        )

    def run(self):
        """ Train an agent """
        log(f"Starting greedy search.")
        log(f"Saving results in:\n\t{self.playground_dir}")

        game = FPGASession(
            design_name=self.design_id,
            design_file=self.design_file,
            playground_dir=self.playground_dir,
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            abc_binary=self.abc_binary,
            ref_abc_seq=self.ref_abc_seq,
            objective=self.objective,
            action_space_id=self.action_space_id,
            max_iterations=self.max_iteration
        )
        assert self.playground_dir == game.playground_dir, (self.playground_dir, game.playground_dir)

        game.reset()
        best_sequence = []
        for step in range(self.max_iteration):
            self.timings.append({})
            # objs = []
            # for action_ind in range(len(self.game.action_space)):
            #     objs.append(self.get_obj(
            #         action_ind=action_ind,
            #         design_file=self.game.get_last_pre_output_design_file()
            #     )
            #     )
            objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
                delayed(self.get_obj)(
                    action=game.action_space[action_ind].act_str,
                    design_file=game.get_last_pre_output_design_file(),
                    iteration=game.iteration,
                    ref_lut_k=game.ref_lut_k,
                    ref_level=game.ref_level

                )
                for action_ind in range(len(game.action_space))
            )
            objs = np.array(objs)
            assert objs.shape == (len(game.action_space),), objs.shape
            best_action_ind = np.random.choice(np.where(objs == objs.min())[0])
            self.log(
                f"{game.iteration}. Best action: {game.action_space[best_action_ind].act_id} "
                f"-> {objs[best_action_ind]}")
            best_sequence.append(best_action_ind)
            game.step(best_action_ind)

        save_w_pickle(game.hist, self.playground_dir, 'hist.pkl')
        save_w_pickle(self.timings, self.playground_dir, 'timing.pkl')
        save_w_pickle(best_sequence, self.playground_dir, 'best_sequence.pkl')
        game.clean()

    def get_obj(self, action: str, design_file: str, iteration: int,
                ref_lut_k: float, ref_level: float):
        """ Return objective value to minimise given an action_id """
        time_ref = time.time()
        sequence = [action]

        try:
            self.log(f"{iteration}. Evaluate {action}")
            obj_lut_k, obj_level, extra_info = get_design_prop(
                seq=sequence, design_file=design_file, mapping=self.mapping,
                library_file='', abc_binary=self.abc_binary,
                lut_inputs=self.lut_inputs, compute_init_stats=False, use_yosys=True
            )
        except CalledProcessError as e:
            if e.args[0] == -6:
                self.log(f"Got error with design: {self.design_id} -> setting objs to refs ")
                obj_lut_k = 10 * ref_lut_k
                obj_level = 10 * ref_level
            else:
                self.log(f"{iteration}. Error {action}")
                raise e
        self.timings[-1][action] = time.time() - time_ref
        return self.objective_function(obj_lut_k / ref_lut_k, obj_level / ref_level)

    def log(self, msg: str, end=None) -> None:
        log(msg, header=f"{self.method_id}-{self.design_id} ({self.seed}) | {self.max_iteration}", end=end)
