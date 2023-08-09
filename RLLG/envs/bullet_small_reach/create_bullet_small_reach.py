# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




try:
    import gym
    import bullet_safety_gym
except ModuleNotFoundError:
    pass
from envs.bullet_small_reach.bullet_small_reach import BulletBallSmallReach
from envs.bullet_small_reach.local_expert_policy import SafeScripted


def create_bullet_small_reach_and_control(orig_cwd='./',
                                         device="cpu"):

    env = BulletBallSmallReach(gym.make('SafetyBallSmallReach-v0'))

    # create controller
    control_dict = {
        "SafeScripted": {
            "coord": None,
            "local_expert": SafeScripted()
        },
    }

    return env, control_dict
