# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




class BulletBallSmallReach:

    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if 'cost_collisions' in info:
            if info['cost_collisions'] >= 0.5:
                # print('catastrophic')
                reward = -1000
                done = True
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def reset(self):
        return self.env.reset()
