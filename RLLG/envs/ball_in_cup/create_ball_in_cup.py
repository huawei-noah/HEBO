# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) Deepmind dm-control.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import dmc2gym
from envs.ball_in_cup.local_expert_policy import SACExpert
import os
from types import MethodType
from typing import Any, Tuple, Dict


def create_ball_in_cup_and_control(orig_cwd: str = './',
                                   device: str = "cpu") -> Tuple[Any, Dict]:
    """
    Create the ball in cup environment and its control (local expert) dictionary.

    Parameters:
    ----------
    orig_cwd : str, optional
        Original current working directory (default is './')
    device : str, optional
        Device (default is 'cpu')

    Returns:
    ----------
    Any
        The ball in cup environment.
    dict
        The control dictionary
    """
    # create env
    env = dmc2gym.make('ball_in_cup', 'catch')

    # modify initialization
    def new_initialize_episode(self, physics: Any) -> None:
        """
        Sets the state of the environment at the start of each episode.

        Parameters:
        ----------
        physics: Any
            An instance of `Physics`
        """
        # Find a collision-free random initial position of the ball.
        penetrating = True
        while penetrating:
            # Assign a random ball position.
            physics.named.data.qpos['ball_x'] = self.random.uniform(-.2, .2)
            physics.named.data.qpos['ball_z'] = self.random.uniform(.0, .25)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        self.after_step(physics)

    try:
        env.env._env._task.initialize_episode = MethodType(new_initialize_episode, env.env._env._task)
    except AttributeError:
        env.env.env._task.initialize_episode = MethodType(new_initialize_episode, env.env.env._task)

    path = os.path.join(orig_cwd, 'envs', 'ball_in_cup', "models")

    control_dict = {
        "MediumSAC": {
            "coord": None,
            "local_expert": SACExpert(env, path, device)
        },
    }

    return env, control_dict
