# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




class LambdaS:

    def __init__(self, pos_tol=None, speed_tol=None):
        self.pos_tol = pos_tol
        self.speed_tol = speed_tol


    def get_use_local(self, env, observation):
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

def ball_in_cup_lambda_s(expert,
                         device="cpu",
                         pos_tol=None,
                         speed_tol=None,
                         smoothed=None):
    return LambdaS()
