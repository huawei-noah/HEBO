# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from typing import Any, Optional
import numpy as np


class SafeScripted:
    """
    SafeScripted class for scripted control.
    """

    def __init__(self) -> None:
        pass

    def get_quarter_position(self, agent: Any, obstacle: Any) -> str:
        """
        Get the quarter position.

        Parameters:
        ----------
        agent : Any
            The agent object.
        obstacle : Any
            The obstacle object.

        Returns:
        ----------
        str
            The quarter position ('below-left', 'below-right', 'above-left', 'above-right').
        """
        pos_x, pos_y = agent.get_position()[:2]
        obstacle_x, obstacle_y = obstacle.get_position()[:2]
        if pos_x <= obstacle_x:
            if pos_y <= obstacle_y:
                return 'below-left'
            return 'above-left'
        if pos_y <= obstacle_y:
            return 'below-right'
        return 'above-right'

    def get_closest_obstacle(self, env: Any) -> int:
        """
        Get the index of the closest obstacle.

        Parameters:
        ----------
        env : Any
            The environment object.

        Returns:
        ----------
        int
            The index of the closest obstacle.
        """
        agent_pos = env.env.env.agent.get_position()[:2]
        pos_los = [obstacle.get_position()[:2] for obstacle in env.env.obstacles]
        return np.argmin(np.linalg.norm(np.vstack(pos_los) - agent_pos, axis=1))

    def get_action(self, observation: np.ndarray, init_action: Optional[Any] = None, env: Optional[Any] = None)\
            -> np.ndarray:
        """
        Get the action for scripted control.

        Parameters:
        ----------
        observation : Any
            The observation.
        init_action : Any, optional
            The initial action (default is None).
        env : Any, optional
            The environment object (default is None).

        Returns:
        ----------
        np.ndarray
            The scripted action.
        """

        # get closest obstacle
        id = self.get_closest_obstacle(env)

        # get quarter for chosen obstacle
        quarter = self.get_quarter_position(env.env.env.agent, env.env.obstacles[id])

        if quarter == 'below-left':
            return np.array([ -0.999, -0.999 ])
        elif quarter == 'below-right':
            return np.array([ 0.999, -0.999 ])
        elif quarter == 'above-left':
            return np.array([ -0.999, 0.999 ])
        elif quarter == 'above-right':
            return np.array([ 0.999, 0.999 ])
