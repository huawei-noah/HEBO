# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.



class LambdaS:

    def __init__(self, pos_tol=0.6):
        self.pos_tol = pos_tol

    def get_use_local(self, env, observation):
        x_pos, y_pos, z_pos = env.env.world.robot_pos()
        if abs(x_pos) > self.pos_tol:
            return 1
        return 0


def point_circle_lambda_s(expert,
                          device="cpu",
                          pos_tol=None,
                          speed_tol=None,
                          smoothed=False
                          ):
    return LambdaS(pos_tol=pos_tol)
