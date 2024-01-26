# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from typing import Union, Any, Dict, List, Optional, Tuple, Callable
from envs.cartpole.create_cartpole import create_cartpole_and_control
from envs.ball_in_cup.create_ball_in_cup import create_ball_in_cup_and_control
from envs.point_mass.create_point_mass import create_point_mass_and_control
from envs.point_circle.create_point_circle import create_point_cirlce_and_control
from envs.bullet_small_reach.create_bullet_small_reach import create_bullet_small_reach_and_control
from envs.hirl_point_fall.create_hirl_point_fall import create_hirl_point_fall_and_control


dict_fn = {
    'cartpole': create_cartpole_and_control,
    'ball_in_cup': create_ball_in_cup_and_control,
    'point_mass': create_point_mass_and_control,
    'point_circle': create_point_cirlce_and_control,
    'bullet_small_reach': create_bullet_small_reach_and_control,
    'hirl_point_fall': create_hirl_point_fall_and_control,
}


def get_env_and_control(name: str = 'ball_in_cup',
                        orig_cwd: str = './',
                        device: str = 'cpu',
                        limit_cart: Optional[float] = 0.6,
                        reward_end: Optional[int] = 1,
                        pos_tol: Optional[float] = 1.) -> Tuple[Any, Dict[Union[str, Tuple[float, float]], Callable]]:
    """
    Returns the environment and local controller.

    Parameters:
    ----------
    name : str, optional
        Name of the environment (default is 'ball_in_cup')
    orig_cwd : str, optional
        Original working directory (default is './')
    device : str, optional
        Device for computation (default is 'cpu')
    limit_cart : float, optional
        Limit for the cart (default is 0.6)
    reward_end : int, optional
        Reward at the end (default is 1)
    pos_tol : float, optional
        Position tolerance (default is 1.0)

    Returns:
    ----------
    Tuple[Any, Dict[Union[str, Tuple[float, float]], Callable]]
        The environment and local controller.
    """
    kwargs = {}

    # get glob name
    if 'pendulum' in name:
        glob_name = 'pendulum'
    elif 'cartpole' in name:
        glob_name = 'cartpole'
        kwargs.update({'limit_cart': limit_cart, 'reward_end': reward_end, 'pos_tol': pos_tol})
    elif 'point_mass' in name:
        glob_name = 'point_mass'
    elif 'hirl_point_fall' in name:
        glob_name = 'hirl_point_fall'
        if 'move_block_only' in name:
            kwargs = {'move_block_only': True}
    else:
        glob_name = name

    if "sparse" in name:
        kwargs.update({'sparse': True})

    if "cartpole" in name:
        kwargs.update({'task_name': name.split('-')[-1]})

    # get env and control
    env, dict_control = dict_fn[glob_name](orig_cwd=orig_cwd,
                                           device=device,
                                           **kwargs)

    return env, dict_control
