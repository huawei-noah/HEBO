# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.



import numpy as np
from dm_control.utils import rewards


class LambdaS:

    def __init__(self, pos_tol=None, speed_tol=None):
        self.pos_tol = pos_tol
        self.speed_tol = speed_tol


    def get_use_local(self, env, observation):
        # check if inside big target or not
        target_size = 0.1                  # env.env.physics.named.model.geom_size['target', 0]
        inside_big_goal = rewards.tolerance(env.env.physics.mass_to_target_dist(),
                                            bounds=(0, target_size))
        if inside_big_goal:
            return 0
        return 1

def point_mass_lambda_s(expert,
                        device="cpu",
                        pos_tol=None,
                        speed_tol=None,
                        smoothed=None):
    return LambdaS()
