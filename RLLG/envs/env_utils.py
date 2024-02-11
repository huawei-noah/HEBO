# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2016 OpenAI (https://openai.com).

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is taken from the gym repository

from typing import Any, Dict, List, Tuple
import gym
import numpy as np


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Normalize action space """

    def __init__(self, env: gym.Env) -> None:
        super(NormalizedEnv, self).__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize the action.

        Parameters:
        ----------
        action : np.ndarray
            The original action

        Returns:
        ----------
        np.ndarray
            The normalized action.
        """
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Reverse the normalized action.

        Parameters:
        ----------
        action : np.ndarray
            The normalized action.

        Returns:
        ----------
        np.ndarray
            The original action.
        """
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class ForcedTimeLimit(gym.Wrapper):
    """
    A wrapper for enforcing a maximum number of steps in an episode.

    Parameters:
    ----------
    env : gym.Env
        The underlying environment.
    max_episode_steps : Optional[int]
        The maximum number of steps in an episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int = None) -> None:
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one timestep of the environment's dynamics, and only return done=True if max_time_steps has been reached.

        Parameters:
        ----------
        action : np.ndarray
            The action to be executed

        Returns:
        ----------
        tuple
            Observation, reward, done, and info.
        """
        observation, reward, done, info = self.env.step(action)
        done = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the environment.

        Returns:
        ----------
        np.ndarray
            The initial observation.
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
