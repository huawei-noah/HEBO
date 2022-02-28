#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 2021.11.10-Add constraint checking
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import re
from subprocess import check_output
from typing import List, Tuple, Optional, Union

import gym
import numpy as np

from DRiLLS.utils import softmax
from utils.utils_misc import log
from .features import extract_features


class SCLSession:
    """
    A class to represent a logic synthesis optimization session using ABC
    """

    def __init__(self, params, design_file: str, playground_dir: str, max_iteration: int):
        self.max_iterations = max_iteration
        self.params = params
        self.playground_dir = playground_dir

        self.design_file = design_file

        self.action_space_length = len(self.params['optimizations'])
        self.observation_space_size = 7 # number of features

        self.iteration = 0
        self.episode = 0
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))

        self.sequence = ['strash']
        self.delay, self.area = float('inf'), float('inf')

        self.best_known_area = (float('inf'), float('inf'), -1, -1)
        self.best_known_delay = (float('inf'), float('inf'), -1, -1)
        self.best_known_area_meets_constraint = (float('inf'), float('inf'), -1, -1)
        self.episode_best_area_meets_constraint = float('inf')

        # logging
        self.log = None

    @property
    def design(self) -> str:
        """ Design name """
        return os.path.basename(self.design_file).split('.')[0]

    def __del__(self):
        if self.log:
            self.log.close()

    def constr_met(self) -> bool:
        """ Whether constraint is met """
        return self.delay <= self.delay_constr

    def reset(self):
        """
        resets the environment and returns the state
        """
        self.iteration = 0
        self.episode += 1
        self.delay, self.area = float('inf'), float('inf')
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))
        self.episode_best_area_meets_constraint = float('inf')
        os.makedirs(self.episode_dir, exist_ok=True)

        # logging
        log_file = os.path.join(self.episode_dir, 'log.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, area, delay, best_area_meets_constraint, best_area, best_delay\n')

        state, _ = self._run()

        # logging
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(self.area), str(self.delay)]) + '\n')
        self.log.flush()

        return state

    def step(self, optimization: int):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][optimization])
        new_state, reward = self._run()
        # logging
        if self.area < self.best_known_area[0]:
            self.best_known_area = (self.area, self.delay, self.episode, self.iteration)
        if self.delay < self.best_known_delay[1]:
            self.best_known_delay = (self.area, self.delay, self.episode, self.iteration)
        if self.constr_met() and self.area < self.best_known_area_meets_constraint[0]:
            self.best_known_area_meets_constraint = (self.area, self.delay, self.episode, self.iteration)
        if self.constr_met() and self.area < self.episode_best_area_meets_constraint:
            self.episode_best_area_meets_constraint = self.area
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(self.area), str(self.delay)]) + ', ' +
                       '; '.join(list(map(str, self.best_known_area_meets_constraint))) + ', ' +
                       '; '.join(list(map(str, self.best_known_area))) + ', ' +
                       '; '.join(list(map(str, self.best_known_delay))) + '\n')
        self.log.flush()

        return new_state, reward, self.iteration == self.max_iterations, {}

    @property
    def library_file(self) -> str:
        return self.params['mapping']['library_file']

    @property
    def abc_binary(self) -> str:
        return self.params['abc_binary']

    @property
    def delay_constr(self) -> float:
        return self.params['mapping']['clock_period']

    def _run(self, with_time_constr: bool = True):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.v')
        pre_output_design_file = os.path.join(self.episode_dir, str(self.iteration - 1) + '.v')
        output_design_file_mapped = os.path.join(self.episode_dir, str(self.iteration) + '-mapped.v')

        abc_command = 'read ' + self.library_file + '; '
        if self.iteration == 1:
            abc_command += 'read ' + self.design_file + '; '
        else:
            abc_command += 'read ' + pre_output_design_file + '; '
        # abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'strash; ' + self.sequence[-1] + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'map'
        if with_time_constr:
            abc_command += ' -D ' + str(self.delay_constr)
        abc_command += '; '
        abc_command += 'write ' + output_design_file_mapped + '; '
        abc_command += 'topo; stime;'

        try:
            proc = check_output([self.abc_binary, '-c', abc_command])
            # get reward
            delay, area = self._get_metrics(proc)
            reward = self._get_reward(delay, area)
            self.delay, self.area = delay, area
            # get new state of the circuit
            state = self._get_state(output_design_file)
            return state, reward
        except Exception as e:
            print(e)
            return None, None

    def get_design_prop_(self, time_constr: Optional[float] = None):
        """ Compute and return delay and area associated to a specific design without changing internal states of
        the game """
        return self.get_design_prop(
            library_file=self.library_file,
            design_file=self.design_file,
            abc_binary=self.abc_binary,
            sequence=self.sequence,
            time_constr=time_constr
        )

    @staticmethod
    def get_design_prop(library_file: str, design_file: str, abc_binary: str, sequence: List[str] = None,
                        time_constr: float = None, verbose: int = 0) -> Tuple[float, float]:
        """
        Compute and return delay and area associated to a specific design

        Args:
            library_file: standard cell library mapping
            design_file: path to the design file
            abc_binary: abc binary path
            sequence: sequence of operations to apply to the design
            time_constr: delay constraint (can be None)
            verbose: verbosity level

        Returns:
            delay, design
        """
        if sequence is None:
            sequence = ['strash; ']
        abc_command = 'read ' + library_file + '; '
        abc_command += 'read ' + design_file + '; '
        abc_command += ';'.join(sequence) + '; '
        abc_command += 'map'
        if time_constr is not None:
            abc_command += ' -D ' + str(time_constr)
        if verbose > 0:
            abc_command += ' -v'
        abc_command += '; '
        abc_command += 'topo; stime;'
        print(abc_command)
        proc = check_output([abc_binary, '-c', abc_command])
        delay, area = SCLSession._get_metrics(proc)
        return delay, area

    @staticmethod
    def _get_metrics(stats):
        """
        parse delay and area from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        ob = re.search(r'Delay *= *[0-9]+.?[0-9]*', line)
        delay = float(ob.group().split('=')[1].strip())
        ob = re.search(r'Area *= *[0-9]+.?[0-9]*', line)
        area = float(ob.group().split('=')[1].strip())

        return delay, area

    def _get_reward(self, delay, area):
        constraint_met = True
        # optimization_improvement: (-1, 0, 1) <=> (worse, same, improvement)
        constraint_improvement = 0  # (-1, 0, 1) <=> (worse, same, improvement)

        # check optimizing parameter
        if area < self.area:
            optimization_improvement = 1
        elif area == self.area:
            optimization_improvement = 0
        else:
            optimization_improvement = -1

        # check constraint parameter
        if not self.constr_met():
            constraint_met = False
            if delay < self.delay:
                constraint_improvement = 1
            elif delay == self.delay:
                constraint_improvement = 0
            else:
                constraint_improvement = -1

        # now calculate the reward
        return self._reward_table(constraint_met, constraint_improvement, optimization_improvement)

    @staticmethod
    def _reward_table(constraint_met, contraint_improvement, optimization_improvement):
        return {
            True: {
                0: {
                    1: 3,
                    0: 0,
                    -1: -1
                }
            },
            False: {
                1: {
                    1: 3,
                    0: 2,
                    -1: 1
                },
                0: {
                    1: 2,
                    0: 0,
                    -1: -2
                },
                -1: {
                    1: -1,
                    0: -2,
                    -1: -3
                }
            }
        }[constraint_met][contraint_improvement][optimization_improvement]

    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.abc_binary)


class SCLSessionEnv(gym.Env):

    def __init__(self, sclsess: SCLSession, normalize_obs: bool = True, softmax_actions: bool = False):
        """
        Gym wrapper for SCLSession

        Args:
            sclsess: original scl session to wrap
            normalize_obs: whether to output normalized observations when calling `step` and `reset`
            softmax_actions: whether the action in `step` method is provided as a single integer or as a vector of
                             arguments on which softmax is applied and actual action is drawn from (not the usual case)
        """
        super(SCLSessionEnv, self).__init__()
        self.sclsess = sclsess
        self.action_space = gym.spaces.Discrete(self.sclsess.action_space_length)
        self.observation_space = gym.spaces.Box(low=-float('inf'),
                                                high=float('inf'),
                                                shape=(self.sclsess.observation_space_size,))
        self.normalize_obs = normalize_obs
        self.normalizer = GymObsNormalizer(self.sclsess.observation_space_size)
        self.softmax_actions = softmax_actions


    def step(self, action: Union[int, np.ndarray]):
        if isinstance(action, np.ndarray):
            assert action.shape == (self.sclsess.action_space_length,), action.shape
            action = np.random.choice(action.shape[0], size=1, p=softmax(action))[0]
        new_state, reward, done, info = self.sclsess.step(action)
        assert new_state[0] > 0
        if self.normalize_obs:
            self.normalizer.observe(new_state)
            new_state = self.normalizer.normalize(new_state)
        if done:
            log(
                f"Agent: {self.agent_learner_id} - Design {self.sclsess.design} | "
                f"Episode {self.sclsess.episode} | "
                f"Iteration {self.sclsess.iteration} | "
                f"Area: {self.sclsess.area} | "
                f"Delay: {self.sclsess.delay} ({'un' if not self.sclsess.constr_met() else ''}met)")
        return new_state, reward, done, info

    @property
    def agent_learner_id(self):
        return os.path.basename(self.sclsess.playground_dir)

    def reset(self):
        state = self.sclsess.reset()
        if self.normalize_obs:
            self.normalizer.reset()
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
        return state

    def render(self, mode='human'):
        pass
