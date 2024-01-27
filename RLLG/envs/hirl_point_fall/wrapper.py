# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2016 OpenAI (https://openai.com).

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import gym
import numpy as np
from typing import Any, Optional, Dict, Tuple


class ForcedTimeLimit(gym.Wrapper):
    """
    A wrapper for enforcing a maximum number of steps in an episode.

    Parameters:
    ----------
    env : gym.Env
        The underlying environment.
    max_episode_steps : Optional[int]
        The maximum number of steps in an episode.
    elapsed_steps : Optional[int]
        The number of steps taken in the current episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Parameters:
        ----------
        action : Any
            The action to be taken.

        Returns:
        ----------
        Tuple
            The observation, reward, done, and info.
        """
        observation, reward, done, info = self.env.step(action)
        done = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs: Any) -> Any:
        """
        Reset the environment.

        Returns:
        ----------
        Any
            The initial observation.
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
