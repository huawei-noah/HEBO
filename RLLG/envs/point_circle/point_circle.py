# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from typing import Any, Tuple, Dict, Optional
import numpy as np


class PointCircle:
    """
    Wrapper for the safe PointCircke environment to change the constraint function into a bad reward.

    Parameters:
    ----------
    env : Any
        The environment to wrap.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        """
        Step through the environment dynamics and change reward function.

        Parameters:
        ----------
        action : Any
            The action to be executed.

        Returns:
        ----------
        tuple
            Observation, reward, done, and info.
        """
        obs, reward, done, info = self.env.step(action)
        if info['cost'] >= 0.5:
            reward = -1000
            done = True
        return obs, reward, done, info

    def render(self, mode: Optional[str] = "human") -> Any:
        """
        Render the environment.

        Parameters:
        ----------
        mode : str, optional
            Rendering mode (default is "human").

        Returns:
        ----------
        Any
            The rendering output.
        """
        return self.env.render(mode)

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        Returns:
        ----------
        Any
            The reset observation.
        """
        return self.env.reset()
