# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Any, List, Optional


class LambdaS:
    """
    Class representing the confidence function.

    Parameters:
    ----------
    pos_tol : float or None, optional
        Position tolerance (default is None)
    speed_tol : float or None, optional
        Speed tolerance (default is None)
    """

    def __init__(self, pos_tol: Optional[float] = None, speed_tol: Optional[float] = None):
        self.pos_tol = pos_tol
        self.speed_tol = speed_tol


    def get_use_local(self, env: Any, observation: List) -> float:
        """
        Get the lambda s value based on the environment and observation.

        Parameters:
        ----------
        env : Any
            The environment
        observation : list of array
            The observation.

        Returns:
        ----------
        float
            Use_local value (0 or 1).
        """
        # check if ball above cup or not, and check if it is inside the cup
        cup_x, cup_z, ball_x, ball_z = observation[0], observation[1], observation[2], observation[3]
        # below cup
        if ball_z <= cup_z + 0.3:
            return 1
        # not inside cup when above cup
        if 0.3 + cup_z <= ball_z <= cup_z + 0.35:
            if ball_x > cup_z + 0.05 or ball_x < cup_z - 0.05:
                return 1
        return 0


def ball_in_cup_lambda_s(expert: Any,
                         device: str = "cpu",
                         pos_tol: float = None,
                         speed_tol: float = None,
                         smoothed: bool = None) -> LambdaS:
    """
    Returns the confience LambdaS instance for the ball-in-cup environment.

    Parameters:
    ----------
    expert : Any
        Expert (not used, but here in case the lambda_s depends on the expert).
    device : str, optional
        Device for computation (default is 'cpu')
    pos_tol : float or None, optional
        Position tolerance (default is None)
    speed_tol : float or None, optional
        Speed tolerance (default is None)
    smoothed : bool or None, optional
        Whether to use smoothed lambda_s (default is None)

    Returns:
    ----------
    LambdaS
        The LambdaS instance
    """
    return LambdaS()
