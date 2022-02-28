# 2021.11.10-refactor imports
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import random
import bisect

from resources.abcRL import PiApprox, BaselineVApprox
from core.algos.GRiLLS.grills_env import EnvGraph


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """

    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value

    def __lt__(self, other):
        return self.value < other.value


class Reinforce(object):
    def __init__(self, env: EnvGraph, gamma: float, pi: PiApprox, baseline: BaselineVApprox, mem_length: int = 4):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.mem_trajectory = []  # the memorized trajectories. sorted by value
        self.mem_length = mem_length
        self.sum_rewards = []

    def gen_trajectory(self, phaseTrain: bool = True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            next_state, reward, term, _ = self._env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = next_state
        return Trajectory(states, rewards, actions, self._env.cur_stats_value())

    def episode(self, phaseTrain=True):
        trajectory = self.gen_trajectory(
            phaseTrain=phaseTrain)  # Generate a trajectory of episode of states, actions, rewards
        self.update_trajectory(trajectory, phaseTrain)
        self._pi.episode()
        return self._env.returns()

    def update_trajectory(self, trajectory, phaseTrain=True):
        states = trajectory.states
        rewards = trajectory.rewards
        actions = trajectory.actions
        bisect.insort(self.mem_trajectory, trajectory)  # memorize this trajectory
        self.len_seq = len(states)  # Length of the episode
        for tIdx in range(self.len_seq):
            G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.len_seq + 1))
            state = states[tIdx]
            action = actions[tIdx]
            baseline = self._baseline(state[0])
            delta = G - baseline
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
        self.sum_rewards.append(sum(rewards))
        # print(sum(rewards))

    def replay(self):
        for idx in range(min(self.mem_length, int(len(self.mem_trajectory) / 10))):
            if len(self.mem_trajectory) / 10 < 1:
                return
            upper = min(len(self.mem_trajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.update_trajectory(self.mem_trajectory[idx])
