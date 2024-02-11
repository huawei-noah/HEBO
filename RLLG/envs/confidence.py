# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from typing import Union, Any, List
import torch
from envs.cartpole.confidence import cartpole_lambda_s
from envs.ball_in_cup.confidence import ball_in_cup_lambda_s
from envs.point_mass.confidence import point_mass_lambda_s
from envs.point_circle.confidence import point_circle_lambda_s
from envs.bullet_small_reach.confidence import bullet_small_reach_lambda_s
from envs.hirl_point_fall.confidence import hirl_point_fall_lambda_s


dict_norm_to_expert = {
    'cartpole': cartpole_lambda_s,
    'ball_in_cup': ball_in_cup_lambda_s,
    'point_mass': point_mass_lambda_s,
    'point_circle': point_circle_lambda_s,
    'hirl_point_fall': hirl_point_fall_lambda_s,
    'bullet_small_reach': bullet_small_reach_lambda_s,
}


def global_lambda_s(glob_name: str,
                    experts: List[torch.nn.Module],
                    device: str = "cpu",
                    pos_tol: Union[float, None] = None,
                    speed_tol: Union[float, None] = None,
                    smoothed: bool = False) -> Any:
    """
    Returns the confidence lambda_s function based on the specified environment type.

    Parameters:
    ----------
    glob_name : str
        Name representing the environment type.
    experts : List[torch.nn.Module]
        List of expert models.
    device : str, optional
        Device for computation (default is 'cpu')
    pos_tol : float or None, optional
        Position tolerance (default is None)
    speed_tol : float or None, optional
        Speed tolerance (default is None)
    smoothed : bool, optional
        Whether to use smoothed lambda_s (default is False)

    Returns:
    ----------
    Any
        The global confidence lambda_s function.
    """
    return dict_norm_to_expert[glob_name](experts,
                                          device=device,
                                          pos_tol=pos_tol,
                                          speed_tol=speed_tol,
                                          smoothed=smoothed
                                          )
