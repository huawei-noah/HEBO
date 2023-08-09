# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




from math import exp


class LambdaS:

    def __init__(self,
                 pos_tol=None,
                 speed_tol=None,
                 smoothed=False):
        self.pos_tol = pos_tol
        self.speed_tol = speed_tol
        self.smoothed = smoothed

    def get_use_local(self, env, observation):
        abs_pos = abs(observation[0])
        if self.smoothed:
            if abs_pos < 0.5:
                return 0
            elif abs_pos > 1.2:
                return 1
            return exp(- 3 * (1.2 - abs_pos))
        else:
            if 1.9 - abs_pos < abs(self.pos_tol):
                return 1
            return 0


def cartpole_lambda_s(expert,
                      device="cpu",
                      pos_tol=None,
                      speed_tol=None,
                      smoothed=False
                      ):
    return LambdaS(pos_tol=pos_tol, speed_tol=speed_tol, smoothed=smoothed)
