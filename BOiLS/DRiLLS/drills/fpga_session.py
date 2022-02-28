#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# 2021.11.10-Add support for more objectives and references
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import time

import gym
import numpy as np
import os
import re
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import shutil
from subprocess import check_output
from typing import List, Tuple, Optional, Union

from DRiLLS.drills.features import extract_features
from DRiLLS.utils import softmax
from core.action_space import Action, ACTION_SPACES
from core.utils.build_in_seq.main import RefObj, BUILD_IN_SEQ
from utils.utils_misc import log


def get_ref(
        design_file: str,
        mapping: str,
        abc_binary: str,
        library_file: str,
        lut_inputs: int,
        ref_abc_seq: str,
        use_yosys: bool
) -> Tuple[float, float]:
    """ Return either area and delay or lut and levels """

    ref_obj = RefObj(design_file=design_file, mapping=mapping, abc_binary=abc_binary,
                     library_file=library_file, lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq,
                     use_yosys=use_yosys)

    ref_1, ref_2 = ref_obj.get_refs()

    return ref_1, ref_2


class FPGASession:
    """
    A class to represent a logic synthesis optimization session using ABC
    """

    def __init__(self, design_name: str, design_file: str, playground_dir: str, action_space_id: str,
                 mapping: str, lut_inputs: int, abc_binary: str, ref_abc_seq: str, objective: str, max_iterations: int):
        """
        Args:
            mapping: either scl of fpga mapping
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """
        # logging
        self.log = None
        self.max_iterations = max_iterations
        self.action_space_id = action_space_id
        self.action_space = ACTION_SPACES[self.action_space_id]
        self.playground_dir = playground_dir

        self.design_name = design_name
        self.design_file = design_file

        assert mapping == 'fpga', f"Mapping should be in fpga, got {mapping}"
        self.mapping = mapping
        self.lut_inputs = lut_inputs
        self.abc_binary = abc_binary

        self.objective = objective
        assert objective in ['both', 'level', 'lut']

        self.action_space_length = len(self.action_space)
        self.ref_abc_seq = ref_abc_seq
        self.observation_space_size = 7  # number of features

        self.iteration = 0
        self.episode = 0
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))

        self.sequence = ['strash']
        self.lut_k, self.level = float('inf'), float('inf')

        self.best_known_lut_k = (float('inf'), float('inf'), -1, -1)
        self.best_known_levels = (float('inf'), float('inf'), -1, -1)
        self.best_known_obj = (float('inf'), float('inf'), -1, -1)  # lut | level | episode | iteration

        init_obj = RefObj(design_file=self.design_file, mapping=self.mapping, abc_binary=self.abc_binary,
                          library_file='', lut_inputs=self.lut_inputs, ref_abc_seq='init', use_yosys=True)

        self.init_lut_k, self.init_level = init_obj.get_refs()

        ref_obj = RefObj(design_file=self.design_file, mapping=self.mapping, abc_binary=self.abc_binary,
                         library_file='', lut_inputs=self.lut_inputs, ref_abc_seq=self.ref_abc_seq, use_yosys=True)

        self.ref_lut_k, self.ref_level = ref_obj.get_refs()
        # self.baseline_impr_lut_k = ((self.init_lut_k - self.ref_lut_k) / self.init_lut_k) / BUILD_IN_SEQ[self.ref_abc_seq].seq_length()
        # self.baseline_impr_level = ((self.init_level - self.ref_level) / self.init_level) / BUILD_IN_SEQ[self.ref_abc_seq].seq_length()
        self.baseline_impr_lut_k = ((self.ref_lut_k - self.init_lut_k) / self.ref_lut_k) / self.max_iterations
        self.baseline_impr_level = ((self.ref_level - self.init_level) / self.ref_level) / self.max_iterations

        self.hist = {
            'init': {
                'lut': self.init_lut_k,
                'level': self.init_level
            },
            'ref': {
                'lut': self.ref_lut_k,
                'level': self.ref_level
            },
            'episodes': []
        }

        log(f"{self.design_name}: {str(self.hist)}")

    def __del__(self):
        if self.log:
            self.log.close()

    def reset(self):
        """
        resets the environment and returns the state
        """
        if os.path.exists(self.episode_dir):
            shutil.rmtree(self.episode_dir)  # remove previous episode dir to save space

        self.iteration = 0
        self.episode += 1
        self.lut_k, self.level = self.init_lut_k, self.init_level
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))
        os.makedirs(self.episode_dir, exist_ok=True)

        # logging
        log_file = os.path.join(self.episode_dir, 'log.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, LUT-k, Levels, best LUT-k / levels, best LUT-k, best levels\n')
        self.hist['episodes'].append([])

        state, _ = self._run()

        # logging
        self.log.write(
            ', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_k)), str(int(self.level))]) + '\n')
        self.log.flush()

        return state

    def clean(self):
        " empty current episode directory"
        if os.path.exists(self.episode_dir):
            shutil.rmtree(self.episode_dir)  # remove previous episode dir to save space

    def step(self, optimization: int):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.action_space[optimization].act_str)
        new_state, reward = self._run()

        # logging
        if self.lut_k < self.best_known_lut_k[0]:
            self.best_known_lut_k = (int(self.lut_k), int(self.level), self.episode, self.iteration)
        if self.level < self.best_known_levels[1]:
            self.best_known_levels = (int(self.lut_k), int(self.level), self.episode, self.iteration)

        self.log.write(
            ', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_k)), str(int(self.level))]) + ', ' +
            '; '.join(list(map(str, ''))) + ', ' +
            '; '.join(list(map(str, self.best_known_lut_k))) + ', ' +
            '; '.join(list(map(str, self.best_known_levels))) + '\n')
        self.log.flush()

        return new_state, reward, self.iteration == self.max_iterations, {}

    def get_last_pre_output_design_file(self):
        return os.path.join(self.episode_dir, str(self.iteration) + '.blif')

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        time_ref = time.time()
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.blif')
        pre_output_design_file = os.path.join(self.episode_dir, str(self.iteration - 1) + '.blif')
        output_design_file_mapped = os.path.join(self.episode_dir, str(self.iteration) + '-mapped.blif')

        abc_command = ''
        if self.iteration == 1:
            abc_command += 'read ' + self.design_file + '; '
        else:
            abc_command += 'read ' + pre_output_design_file + '; '
        # abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'strash; ' + self.sequence[-1] + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'if -K ' + str(self.lut_inputs) + '; '
        abc_command += 'write ' + output_design_file_mapped + '; '
        abc_command += 'print_stats;'

        try:
            proc = check_output([self.abc_binary, '-c', abc_command])
            # get reward
            new_lut_k, new_levels = self._get_metrics(proc)
            reward = self._get_reward(new_lut_k, new_levels)
            self.lut_k, self.level = new_lut_k, new_levels
            # get new state of the circuit
            state = self._get_state(output_design_file)
            self.hist['episodes'][-1].append({'level': self.level, 'lut': self.lut_k, 'action': self.sequence[-1],
                                              'time': time.time() - time_ref})

            return state, reward
        except Exception as e:
            raise
            # return None, None

    def get_design_prop_(self, lut_inputs: int, verbose: Optional[int] = 0):
        """ Compute and return lutk and levels associated to a specific design without changing internal states of
        the game """
        return self.get_design_prop(
            library_file='',
            design_file=self.design_file,
            abc_binary=self.abc_binary,
            sequence=self.sequence,
            verbose=verbose,
            lut_inputs=lut_inputs
        )

    @staticmethod
    def get_design_prop(library_file: str, design_file: str, abc_binary: str, lut_inputs: int,
                        sequence: List[str] = None, verbose: Optional[int] = 0) -> Tuple[int, int]:
        """
        Compute and return delay and area associated to a specific design

        Args:
            library_file: standard cell library mapping
            design_file: path to the design file
            abc_binary: abc binary path
            sequence: sequence of operations to apply to the design
            lut_inputs: number of LUT inputs (2 < num < 33)
            verbose: verbosity level

        Returns:
            lut_K, levels
        """
        if sequence is None:
            sequence = []
        new_sequence = []
        for action in sequence:
            if 'strash' not in action:
                new_sequence.append('strash; ')
            new_sequence.append(action)
        if len(sequence) == 0:
            new_sequence = ['strash; ']
        sequence = new_sequence
        abc_command = 'read ' + library_file + '; '
        abc_command += 'read ' + design_file + '; '
        abc_command += ';'.join(sequence) + '; '
        abc_command += f"if {'-v ' if verbose > 0 else ''}-K {lut_inputs}; "
        abc_command += 'print_stats; '
        # print(abc_command)
        proc = check_output([abc_binary, '-c', abc_command])
        try:
            lut_K, levels = FPGASession._get_metrics(proc)
        except AttributeError as e:
            raise AttributeError(f'Problem to extract stats from {design_file}\n\t{abc_command}\n') from e
        # print(lut_K, levels)
        return lut_K, levels

    @staticmethod
    def _get_metrics(stats) -> Tuple[int, int]:
        """
        parse LUT count and levels from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()

        ob = re.search(r'lev *= *[0-9]+', line)
        levels = int(ob.group().split('=')[1].strip())

        ob = re.search(r'nd *= *[0-9]+', line)
        lut_k = int(ob.group().split('=')[1].strip())

        return lut_k, levels

    # def _get_reward(self, new_lut_k: int, new_levels: int):
    #     rel_impr_lut = (self.lut_k - new_lut_k) / self.init_lut_k - self.baseline_impr_lut_k
    #     rel_impr_level = (self.level - new_levels) / self.init_level - self.baseline_impr_level
    #     if self.objective == 'both':
    #         return (rel_impr_level + rel_impr_lut) / 2
    #     if self.objective == 'lut':
    #         return rel_impr_lut
    #     if self.objective == 'level':
    #         return rel_impr_level
    #     raise ValueError(self.objective)
    #
    def _get_reward(self, new_lut_k: int, new_levels: int):
        rel_impr_lut = (self.lut_k - new_lut_k) / self.ref_lut_k - self.baseline_impr_lut_k
        rel_impr_level = (self.level - new_levels) / self.ref_level - self.baseline_impr_level
        if self.objective == 'both':
            return (rel_impr_level + rel_impr_lut) / 2
        if self.objective == 'lut':
            return rel_impr_lut
        if self.objective == 'level':
            return rel_impr_level
        raise ValueError(self.objective)

    def _get_state(self, design_file):
        return extract_features(
            design_file=design_file,
            yosys_binary='yosys',
            abc_binary=self.abc_binary
        )


class GymObsNormalizer:
    n: np.ndarray
    mean: np.ndarray
    mean_diff: np.ndarray
    var: np.ndarray

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.reset()

    def observe(self, x):
        self.n += 1.
        last_mean = np.copy(self.mean)
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = np.clip(self.mean_diff / self.n, a_min=1e-2, a_max=1000000000)

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def reset(self):
        self.n = np.zeros(self.num_inputs)
        self.mean = np.zeros(self.num_inputs)
        self.mean_diff = np.zeros(self.num_inputs)
        self.var = np.zeros(self.num_inputs)


class FPGASessionEnv(gym.Env):

    def __init__(self, fpgasess: FPGASession, normalize_obs: bool = True, softmax_actions: bool = False):
        """
        Gym wrapper for FPGASession

        Args:
            fpgasess: original fpga session to wrap
            normalize_obs: whether to output normalized observations when calling `step` and `reset`
            softmax_actions: whether the action in `step` method is provided as a single integer or as a vector of
                             arguments on which softmax is applied and actual action is drawn from (not the usual case)
        """
        super(FPGASessionEnv, self).__init__()
        self.fpgasess: FPGASession = fpgasess
        self.action_space = gym.spaces.Discrete(self.fpgasess.action_space_length)
        self.observation_space = gym.spaces.Box(low=-float('inf'),
                                                high=float('inf'),
                                                shape=(self.fpgasess.observation_space_size,))
        self.normalize_obs = normalize_obs
        self.normalizer = GymObsNormalizer(self.fpgasess.observation_space_size)
        self.softmax_actions = softmax_actions

    def step(self, action: Union[int, np.ndarray]):
        if isinstance(action, np.ndarray):
            assert action.shape == (self.fpgasess.action_space_length,), action.shape
            action = np.random.choice(action.shape[0], size=1, p=softmax(action))[0]
        new_state, reward, done, info = self.fpgasess.step(action)
        assert new_state[0] > 0
        if self.normalize_obs:
            self.normalizer.observe(new_state)
            new_state = self.normalizer.normalize(new_state)
        if done:
            log(
                f"Agent: {self.agent_learner_id} - Design {self.fpgasess.design_name} | "
                f"Episode {self.fpgasess.episode} | "
                f"Iteration {self.fpgasess.iteration} | "
                f"Lut-k: {self.fpgasess.lut_k} | "
                f"Levels: {self.fpgasess.level}")
        return new_state, reward, done, info

    @property
    def agent_learner_id(self):
        return os.path.basename(self.fpgasess.playground_dir)

    def reset(self):
        state = self.fpgasess.reset()
        if self.normalize_obs:
            self.normalizer.reset()
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
        return state

    def render(self, mode='human'):
        pass
