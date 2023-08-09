# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 dm-control (https://github.com/deepmind/dm_control).

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The initialization and termination function of the environment has been slightly changed from the original one.


import dmc2gym
from envs.cartpole.local_expert_policy import SafeScripted
import os
from types import MethodType

def new_get_termination(limit_cart=0.6):
    def new_get_termination_fn(self, physics):
        pos = physics.named.data.qpos['slider'][0]
        if abs(pos) > limit_cart:
            return 1
    return new_get_termination_fn

def new_get_reward(limit_cart=0.6, reward_end=1):
    def new_get_reward_fn(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        pos = physics.named.data.qpos['slider'][0]
        if abs(pos) > limit_cart:
            return -reward_end
        return self._get_reward(physics, sparse=self._sparse)
    return new_get_reward_fn


def create_cartpole_and_control(orig_cwd='./',
                                device="cpu",
                                task_name="swingup",
                                limit_cart=0.6,
                                reward_end=1,
                                pos_tol=1.):

    # create env
    env = dmc2gym.make(domain_name="cartpole", task_name=task_name)

    # change termination and reward function
    env.env.task.get_termination = MethodType(new_get_termination(limit_cart=limit_cart), env.env.task)
    env.env.task.get_reward = MethodType(new_get_reward(limit_cart=limit_cart,
                                                        reward_end=reward_end), env.env.task)

    # create controller
    path = os.path.join(orig_cwd, 'envs', 'cartpole', "models")
    control_dict = {
        "SafeScripted": {
            "coord": None,
            "local_expert": SafeScripted()
        },
    }

    return env, control_dict
