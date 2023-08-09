# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




from envs.hirl_point_fall.point_fall import PointFallEnv
from envs.hirl_point_fall.local_expert_policy import SACExpert
from envs.hirl_point_fall.wrapper import ForcedTimeLimit
from envs.env_utils import NormalizedEnv
import os


def create_hirl_point_fall_and_control(move_block_only=False,
                                       orig_cwd='./',
                                       device="cpu"):

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
