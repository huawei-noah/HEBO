# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




try:
    from safety_gym.envs.engine import Engine
except ModuleNotFoundError:
    pass
from typing import Any, Tuple, Dict
from envs.point_circle.point_circle import PointCircle
from envs.point_circle.local_expert_policy import SafeScripted
import os


def create_point_cirlce_and_control(orig_cwd: str ='./',
                                    device: str ="cpu") -> Tuple[Any, Dict]:
    """
    Create the Point Circle environment and its associated controller.

    Parameters:
    ----------
    orig_cwd : str, optional
        Original current working directory (default is './')
    device : str, optional
        Device to run the environment on (default is "cpu")

    Returns:
    ----------
    Tuple[Any, Dict[str, Any]]
        Tuple containing the environment and the controller dictionary.
    """
    config_dict = {
        'robot_base': 'xmls/point.xml',
        'task': 'circle',
        'observe_goal_lidar': False,
        'observe_box_lidar': False,
        'observe_circle': True,
        'lidar_max_dist': 6
    }
    init_env = Engine(config=config_dict)
    env = PointCircle(init_env)

    # create controller
    control_dict = {
        "SafeScripted": {
            "coord": None,
            "local_expert": SafeScripted()
        },
    }

    return env, control_dict
