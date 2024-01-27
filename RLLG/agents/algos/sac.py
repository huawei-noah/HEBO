# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any, Dict, List, Union
from ml_collections import ConfigDict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.common.model import Scalar, soft_target_update, SamplerPolicy


class SAC(object):
    """
    Soft Actor-Critic (SAC) algorithm implementation.

    Parameters:
    -----------
    config: dict
        Configuration parameters for SAC.
    policy: torch.nn.Module
        The policy network.
    sampler_policy: SamplerPolicy
        The sampler policy network.
    qf1: torch.nn.Module
        The first critic network.
    qf2: torch.nn.Module
        The second critic network.
    target_qf1: torch.nn.Module
        The target network for the first critic.
    target_qf2: torch.nn.Module
        The target network for the second critic.
    """

    @staticmethod
    def get_default_config(updates: Optional[Dict] = None) -> ConfigDict:
        """
        Get the default configuration for SAC.

        Parameters:
        -----------
        updates: dict, optional
            Optional dictionary to update default configuration.

        Returns:
        --------
        ConfigDict
            Default configuration for SAC.
        """
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 1.0
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = True
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self,
                 config: Dict,
                 policy: torch.nn.Module,
                 sampler_policy: SamplerPolicy,
                 qf1: torch.nn.Module,
                 qf2: torch.nn.Module,
                 target_qf1: torch.nn.Module,
                 target_qf2: torch.nn.Module):
        self.config = SAC.get_default_config(config)
        self.policy = policy
        self.sampler_policy = sampler_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        optimizer_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate: float) -> None:
        """
        Update the target networks with soft target updates.

        Parameters:
        -----------
        soft_target_update_rate: float
            Rate of soft target network updates.
        """
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch: Dict[str, Any], batch_success: Optional[Dict[str, torch.Tensor]] = None) -> Dict[
        str, Any]:
        """
        Train the SAC (Soft Actor-Critic) agent on a batch of experiences.

        Parameters:
        -----------
        batch: dict
            A dictionary containing the the transitions.
        batch_success: dict, optional
            A dictionary containing the the transitions.

        Returns:
        --------
        dict
            A dictionary containing training metrics.
        """
        self._total_steps += 1

        # classic obs
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        new_actions, log_pi = self.policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        q_new_actions = torch.min(
            self.qf1(observations, new_actions),
            self.qf2(observations, new_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        with torch.no_grad():
            new_next_actions, next_log_pi = self.policy(next_observations)

            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

        q_target = self.config.reward_scale * rewards + (1. - dones) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        metrics_to_return = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        return metrics_to_return

    def torch_to_device(self, device: torch.device) -> None:
        """
        Move all modules to the specified device.

        Parameters:
        -----------
        device: torch.device
            The target device.
        """
        for module in self.modules:
            module.to(device)

    def get_action(self,
                   env: Any,
                   observation: np.ndarray,
                   deterministic: bool = False,
                   add_local_information: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, np.ndarray]]:
        """
        Get an action from the policy.

        Parameters:
        -----------
        env: Any
            The environment.
        observation: np.ndarray
            The current observation.
        deterministic: bool, optional
            Whether to sample a deterministic action.
        add_local_information: bool, optional
            Whether to add local information.

        Returns:
        --------
        Tuple[np.ndarray, float, np.ndarray]
            The action, local information, and expert action.
        """
        action = self.sampler_policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
        if add_local_information:
            return action, 0, np.zeros(action.shape)
        return action

    @property
    def modules(self) -> List[torch.nn.Module]:
        """
        Get a list of modules.

        Returns:
        --------
        List[nn.Module]
            The list of modules including policy, q-functions, and optional log_alpha.
        """
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        return modules

    @property
    def total_steps(self) -> int:
        """
        Get the total number of steps taken.

        Returns:
        --------
        int
            The total number of steps.
        """
        return self._total_steps
