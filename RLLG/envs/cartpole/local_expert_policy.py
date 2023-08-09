# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




import numpy as np


class SafeScripted:

    def __init__(self):
        pass

    def get_action(self, observation, init_action=None):
        pos = observation[0]
        if pos > 0:
            return np.float32(np.array([-0.999]))
        return np.float32(np.array([0.999]))
