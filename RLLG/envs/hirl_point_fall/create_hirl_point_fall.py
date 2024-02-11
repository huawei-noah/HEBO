# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




from typing import Any, Tuple, Dict
from envs.hirl_point_fall.point_fall import PointFallEnv
from envs.hirl_point_fall.local_expert_policy import SACExpert
from envs.hirl_point_fall.wrapper import ForcedTimeLimit
from envs.env_utils import NormalizedEnv
import os


def create_hirl_point_fall_and_control(move_block_only: bool = False,
                                       orig_cwd: str = './',
                                       device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """
    Create the Point Fall environment and its associated controller.

    Parameters:
    ----------
    move_block_only : bool, optional
        If True, move only the block; if False, move both the block and the robot (default is False)
    orig_cwd : str, optional
        Original current working directory (default is './')
    device : str, optional
        Device to run the environment on (default is "cpu")

    Returns:
    ----------
    Tuple[An, Dict[str, Any]]
        Tuple containing the environment and the controller dictionary.
    """

    init_env = PointFallEnv(move_block_only=move_block_only, scaling_factor=4, max_steps=1000)
    env = ForcedTimeLimit(NormalizedEnv(init_env), max_episode_steps=1000)

    path = os.path.join(orig_cwd, 'envs', 'hirl_point_fall', "models")

    # create controller
    control_dict = {
        "MediumSAC": {
            "coord": None,
            "local_expert": SACExpert(env, path, device)
        },
    }

    return env, control_dict
