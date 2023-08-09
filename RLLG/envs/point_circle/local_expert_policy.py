# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




import numpy as np
import torch
import os


class SafeScripted:

    def __init__(self):
        pass

    def get_action(self, observation, init_action=None, env=None):
        x_pos, y_pos, z_pos = env.env.world.robot_pos()
        rot_mat = env.env.world.robot_mat()
        theta = np.arctan2(-rot_mat[0, 1], rot_mat[0, 0])
        if x_pos > 0:
            if abs(theta) >= 3 * np.pi / 4:
                if y_pos > 0 and theta > 0:
                    return np.array([0.999, 0.5])
                elif y_pos < 0 and theta < 0:
                    return np.array([0.999, -0.5])
                else:
                    return np.array([0.999, 0])
            elif theta < 0:
                return np.array([-0.999, -0.999])
            else:
                return np.array([-0.999, 0.999])
        else:
            if abs(theta) <= np.pi / 4:
                if y_pos > 0 and theta > 0:
                    return np.array([0.999, -0.5])
                elif y_pos < 0 and theta < 0:
                    return np.array([0.999, 0.5])
                else:
                    return np.array([0.999, 0])
            elif theta < 0:
                return np.array([-0.999, 0.999])
            else:
                return np.array([-0.999, -0.999])
