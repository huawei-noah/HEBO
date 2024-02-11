# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.



from typing import Any, Optional
import numpy as np
import torch
import os


class SACExpert:
    """
    Soft Actor-Critic (SAC) Expert.

    Parameters:
    ----------
    env : Any
        The environment (usually dm control env, could be gym as well or others).
    path : str
        The path to the model.
    device : str, optional
        The device to run the expert policy (default is 'cpu').
    """

    def __init__(self, env: Any, path: str, device: Optional[str] = "cpu") -> None:
        from agents.common.model import TanhGaussianPolicy, SamplerPolicy
        # hyper-params
        policy_arch = '64-64'
        policy_log_std_multiplier = 1.0
        policy_log_std_offset = -1.0

        # load expert policy
        expert_policy = TanhGaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            policy_arch,
            log_std_multiplier=policy_log_std_multiplier,
            log_std_offset=policy_log_std_offset,
        )
        glob_path = os.path.join(path, 'medium_expert_sac')

        expert_policy.load_state_dict(torch.load(glob_path))
        expert_policy.to(device)
        self.sampling_expert_policy = SamplerPolicy(expert_policy, device=device)

    def get_action(self, observation: np.ndarray, init_action: Any = None, env: Any = None) -> np.ndarray:
        """
        Get an action from the SAC expert policy.

        Parameters:
        ----------
        observation : numpy.ndarray
            The observation from the environment.
        init_action : Any, optional
            Initial action (default is None).
        env : gym.Env, optional
            The environment (default is None).

        Returns:
        ----------
        numpy.ndarray
            The clipped expert action.
        """
        with torch.no_grad():
            expert_action = self.sampling_expert_policy(
                np.expand_dims(observation, 0), deterministic=True
            )[0, :]
        return np.clip(expert_action, a_min=-0.99, a_max=0.99)  # expert_action
