# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




import numpy as np


class LambdaS:

    def __init__(self, pos_tol=3.):
        self.pos_tol = pos_tol

    def get_use_local(self, env, observation):
        agent_pos = env.env.env.agent.get_position()[:2]
        pos_los = [obstacle.get_position()[:2] for obstacle in env.env.obstacles]
        min_distance = np.min(np.linalg.norm(np.vstack(pos_los) - agent_pos, axis=1))
        if abs(min_distance) <= self.pos_tol:
            return 1
        return 0


def bullet_small_reach_lambda_s(expert,
                               device="cpu",
                               pos_tol=None,
                               speed_tol=None,
                               smoothed=False
                               ):
    return LambdaS()
