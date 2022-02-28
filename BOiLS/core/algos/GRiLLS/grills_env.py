# 2021.11.10-modified the reward function
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
from typing import List

import abc_py
import numpy as np
import torch
from dgl import DGLGraph

from resources.abcRL.graphExtractor import extract_dgl_graph
from core.action_space import Action
from core.sessions.utils import get_design_prop
from core.utils.build_in_seq.main import RefObj


class Stats:
    """ Auxiliary class to store stats (area & delay / luts & levels)"""

    def __init__(self, obj_1: float, obj_2: float):
        self.obj_1 = obj_1
        self.obj_2 = obj_2

    def copy(self):
        return Stats(self.obj_1, self.obj_2)


class EpisodeHist:

    def __init__(self):
        self.action_seq: List[str] = []
        self.action_ind_seq: List[int] = []
        self.objs_1: List[float] = []
        self.objs_2: List[float] = []


class EnvGraph:
    """
    Environment for GRiLLS
    """

    def __init__(self, design_file: str, library_file: str, abc_binary: str, mapping: str, lut_inputs: int,
                 objective: str, use_yosys: bool,
                 seq_length: int, ref_obj_1: float, ref_obj_2: float, action_space: List[Action], playground_path: str):

        self.design_file = design_file
        self.current_design_file = design_file

        self.library_file = library_file
        self.abc_binary = abc_binary
        self.mapping = mapping
        self.lut_inputs = lut_inputs

        self.seq_length = seq_length

        self.ref_obj_1 = ref_obj_1
        self.ref_obj_2 = ref_obj_2

        self.action_space = action_space

        self.episode = 0
        self.episode_hist: EpisodeHist = EpisodeHist()

        self.playground_path = playground_path

        assert objective in ['both']
        self.objective = objective
        self.use_yosys = use_yosys

        self._abc = abc_py.AbcInterface()
        # self._abc.start()
        # self._abc.read(self.design_file)

        # get initial stats
        init_obj = RefObj(design_file=self.design_file, mapping=self.mapping, abc_binary=self.abc_binary,
                          library_file='', lut_inputs=self.lut_inputs, ref_abc_seq='init', use_yosys=True)

        self.init_obj_1, self.init_obj_2 = init_obj.get_refs()

        self.current_len_seq = 0

        # QoR_init - QoR_ref
        self.baseline_impr_lut_k = ((self.ref_obj_1 - self.init_obj_1) / self.ref_obj_1) / self.seq_length
        self.baseline_impr_level = ((self.ref_obj_2 - self.init_obj_1) / self.ref_obj_2) / self.seq_length

        # self.ref_reward = (self.init_obj_1 / self.ref_obj_1 + self.init_obj_2 / self.ref_obj_2) - 2
        # self.ref_reward_baseline = self.ref_reward / self.seq_length  # mimic reward obtained after each action
        # log(f"Ref QoR improvement for {os.path.basename(self.design_file)}: {self.ref_reward:.2f}")

        self.last_stats: Stats = Stats(self.init_obj_1, self.init_obj_2)  # The initial AIG statistics
        self.cur_stats: Stats = Stats(self.init_obj_1, self.init_obj_2)  # the current AIG statistics
        self.last_act = self.num_actions - 1
        self.last_act2 = self.num_actions - 1
        self.last_act3 = self.num_actions - 1
        self.last_act4 = self.num_actions - 1

    def reset(self):
        self.current_len_seq = 0
        self._abc.end()
        self._abc.start()
        # Delete current design file and restore to initial design file
        if self.current_design_file != self.design_file:
            assert self.episode > 0, self.episode
            os.remove(self.current_design_file)
            os.rmdir(self.episode_playground_path)
        self.current_design_file = self.design_file
        self._abc.read(self.current_design_file)
        self.episode_hist = EpisodeHist()

        self.last_stats: Stats = Stats(self.init_obj_1, self.init_obj_2)  # The initial AIG statistics
        self.cur_stats: Stats = Stats(self.init_obj_1, self.init_obj_2)  # the current AIG statistics
        self.last_act = self.num_actions - 1
        self.last_act2 = self.num_actions - 1
        self.last_act3 = self.num_actions - 1
        self.last_act4 = self.num_actions - 1
        self.episode += 1
        os.makedirs(self.episode_playground_path, exist_ok=True)
        return self.state()

    @property
    def episode_playground_path(self) -> str:
        return os.path.join(self.playground_path, f"ep-{self.episode}")

    def close(self):
        self.reset()

    def step(self, action_id):
        self.take_action(action_id)
        self._abc.read(self.current_design_file)
        next_state = self.state()
        reward = self.reward()
        done = self.current_len_seq >= self.seq_length
        return next_state, reward, done, 0

    def take_action(self, action_id) -> None:
        """
        Take action
        """
        self.last_act4 = self.last_act3
        self.last_act3 = self.last_act2
        self.last_act2 = self.last_act
        self.last_act = action_id

        self.current_len_seq += 1

        assert action_id < self.num_actions, (action_id, self.num_actions)
        sequence = [self.action_space[action_id].act_id if not self.use_yosys else self.action_space[action_id].act_str]

        new_design_filepath = os.path.join(self.episode_playground_path, 'out.blif')
        obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=self.current_design_file,
                                                   mapping=self.mapping, compute_init_stats=False,
                                                   library_file=self.library_file, abc_binary=self.abc_binary,
                                                   lut_inputs=self.lut_inputs,
                                                   write_unmap_design_path=new_design_filepath,
                                                   use_yosys=self.use_yosys)
        self.current_design_file = new_design_filepath

        # update episode hist
        self.episode_hist.action_seq.append(self.action_space[action_id].act_id)
        self.episode_hist.action_ind_seq.append(action_id)
        self.episode_hist.objs_1.append(obj_1)
        self.episode_hist.objs_2.append(obj_2)

        # update the statitics
        self.last_stats: Stats = self.cur_stats.copy()
        self.cur_stats: Stats = Stats(obj_1=obj_1, obj_2=obj_2)

    def state(self) -> [torch.Tensor, DGLGraph]:
        """
        Current state
        """
        one_hot_act = np.zeros(self.num_actions + 1)  # account for end action
        one_hot_act[self.last_act] = 1
        last_one_hot_acts = np.zeros(self.num_actions + 1)
        last_one_hot_acts[self.last_act2] += 1 / 3
        last_one_hot_acts[self.last_act3] += 1 / 3
        last_one_hot_acts[self.last_act4] += 1 / 3
        state_array = np.array([
            self.cur_stats.obj_1 / self.init_obj_1,
            self.cur_stats.obj_2 / self.init_obj_2,
            self.last_stats.obj_1 / self.init_obj_1,
            self.last_stats.obj_2 / self.init_obj_2,
        ])
        step_array = np.array([float(self.current_len_seq) / self.seq_length])
        combined = np.concatenate((state_array, last_one_hot_acts, step_array), axis=-1)
        combined_torch = torch.from_numpy(combined.astype(np.float32)).float()
        graph: DGLGraph = extract_dgl_graph(self._abc)
        return combined_torch, graph

    def reward(self):
        assert self.last_act < self.num_actions
        rel_impr_lut = (self.last_stats.obj_1 - self.cur_stats.obj_1) / self.ref_obj_1 - self.baseline_impr_lut_k
        rel_impr_level = (self.last_stats.obj_2 - self.cur_stats.obj_2) / self.ref_obj_2 - self.baseline_impr_level
        if self.objective == 'both':
            return (rel_impr_level + rel_impr_lut) / 2
        if self.objective == 'lut':
            return rel_impr_lut
        if self.objective == 'level':
            return rel_impr_level
        raise ValueError(self.objective)

    @property
    def num_actions(self):
        return len(self.action_space)

    def dim_state(self):
        """
        State dimension:
            - 4: current/latest nb nodes/levels
            - num_actions + 1: encoding of previous actions
            - 1: scalar representing current step
        """
        return 4 + (self.num_actions + 1) + 1

    def returns(self):
        pass
        # return [self.cur_stats.numAnd, self.cur_stats.lev]

    def stat_value(self, stat: Stats) -> float:
        """ QoR """
        return stat.obj_1 / self.ref_obj_1 + stat.obj_2 / self.ref_obj_2

    def seed(self, sd):
        pass

    def cur_stats_value(self):
        return self.stat_value(self.cur_stats)
