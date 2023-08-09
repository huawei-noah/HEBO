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

    def get_quarter_position(self, agent, obstacle):
        pos_x, pos_y = agent.get_position()[:2]
        obstacle_x, obstacle_y = obstacle.get_position()[:2]
        if pos_x <= obstacle_x:
            if pos_y <= obstacle_y:
                return 'below-left'
            return 'above-left'
        if pos_y <= obstacle_y:
            return 'below-right'
        return 'above-right'

    def get_closest_obstacle(self, env):
        agent_pos = env.env.env.agent.get_position()[:2]
        pos_los = [obstacle.get_position()[:2] for obstacle in env.env.obstacles]
        return np.argmin(np.linalg.norm(np.vstack(pos_los) - agent_pos, axis=1))

    def get_action(self, observation, init_action=None, env=None):

        # get closest obstacle
        id = self.get_closest_obstacle(env)

        # get quarter for chosen obstacle
        quarter = self.get_quarter_position(env.env.env.agent, env.env.obstacles[id])

        if quarter == 'below-left':
            return np.array([ -0.999, -0.999 ])
        elif quarter == 'below-right':
            return np.array([ 0.999, -0.999 ])
        elif quarter == 'above-left':
            return np.array([ -0.999, 0.999 ])
        elif quarter == 'above-right':
            return np.array([ 0.999, 0.999 ])
