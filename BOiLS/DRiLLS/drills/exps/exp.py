# 2021.11.10-add support of multiple objectives and add classes for type of solver
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

import abc
import os
from typing import List

from DRiLLS.drills.models.agent import Agent
from core.design_groups import get_designs_path
from utils.utils_misc import log
from utils.utils_save import get_storage_root


def obj_both(ratio_1, ratio_2):
    return ratio_1 + ratio_2


def obj_level(ratio_1, ratio_2):
    return ratio_2


def obj_lut(ratio_1, ratio_2):
    return ratio_1


def obj_min_improvements(ratio_1, ratio_2):
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


class ExpABC(abc.ABC):
    playground_dir: str
    abc_binary = 'yosys-abc'

    def __init__(
            self, design_id: str, max_iteration: int, mapping: str, lut_inputs: int,
            ref_abc_seq: str,
            objective: str, action_space_id: str, seed: int,
    ):
        self.design_id = design_id
        self.design_file = get_designs_path(self.design_id)
        assert len(self.design_file) == 1, self.design_file
        self.design_file = self.design_file[0]
        self.ref_abc_seq = ref_abc_seq
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

        self.action_space_id = action_space_id
        self.seed = seed

        self.max_iteration = max_iteration

        self.mapping = mapping
        self.lut_inputs = lut_inputs

    @staticmethod
    def already_trained(playground_dir: str) -> bool:
        end_result = os.path.join(playground_dir, f"hist.pkl")
        return os.path.exists(end_result)

    def already_trained_(self) -> bool:
        return self.already_trained(
            playground_dir=self.playground_dir,
        )

    @staticmethod
    def get_exp_id_basis(action_space_id: str, lut_inputs: int, objective: str, ref_abc_seq: str, seq_length: int):
        base_id = f"act-space-{action_space_id}_lut-{lut_inputs}_obj-{objective}_ref-seq-{ref_abc_seq}"
        if seq_length != 10:
            base_id += f"seq-{seq_length}"
        return base_id


class RLExp(ExpABC, abc.ABC):
    learner: Agent

    def __init__(self, design_id: str, episodes: int, max_iteration, mapping: str, lut_inputs: int, ref_abc_seq: str,
                 objective: str, action_space_id: str, seed: int, load_model_from: str = None):
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
        self.episodes = episodes
        self.load_model_from = load_model_from

    @staticmethod
    def get_playground_dir(
            design,
            exp_id,
            learner_id,
            seed,
    ) -> str:
        playground_dir = os.path.join(get_storage_root(), 'RL', 'drills',
                                      'playground', design, exp_id, learner_id, str(seed))
        return playground_dir

    @abc.abstractmethod
    def train(self):
        """ Train an agent """
        log(f"Starting to train the agent on {self.design_id}")
        log(f"Saving results in:\n\t{self.playground_dir}")
        pass

    @abc.abstractmethod
    def optimize(self, design_ids: List[str], max_iterations: int, overwrite: bool):
        pass

    @property
    def exp_id(self) -> str:
        return self.get_exp_id(
            action_space_id=self.action_space_id,
            lut_inputs=self.lut_inputs,
            objective=self.objective,
            ref_abc_seq=self.ref_abc_seq,
            seq_length=self.max_iteration,
            n_episodes=self.episodes
        )

    @staticmethod
    def get_exp_id(action_space_id: str, lut_inputs: int, objective: str, ref_abc_seq: str, seq_length: int,
                   n_episodes: int):
        base_id = ExpABC.get_exp_id_basis(
            action_space_id=action_space_id,
            lut_inputs=lut_inputs,
            objective=objective,
            ref_abc_seq=ref_abc_seq,
            seq_length=seq_length
        )
        if n_episodes != 100:
            base_id += f"n-ep-{n_episodes}"
        return base_id

    def get_model_path(self) -> str:
        """ Path where model checkpoints will be stored """
        return self.learner.model_path()
