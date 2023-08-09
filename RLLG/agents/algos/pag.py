# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from ml_collections import ConfigDict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.common.model import Scalar, soft_target_update


class PAG(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 1.0
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.use_automatic_entropy_tuning_parametrized_perturbation = True
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

    def __init__(self, config, policy, sampler_policy, qf1, qf2, target_qf1, target_qf2,
                 use_local, local_expert, parametrized_perturbation, sampler_parametrized_perturbation):
        self.config = PAG.get_default_config(config)
        self.policy = policy
        self.sampler_policy = sampler_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.parametrized_perturbation = parametrized_perturbation
        self.sampler_parametrized_perturbation = sampler_parametrized_perturbation

        # hyperparams
        self.use_local = use_local
        self.beta = 1.
        self.local_expert = local_expert

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
        self.parametrized_perturbation_optimizer = optimizer_class(
            self.parametrized_perturbation.parameters(), self.config.policy_lr,
        )

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.config.use_automatic_entropy_tuning_parametrized_perturbation:
            self.expert_log_alpha = Scalar(0.0)
            self.expert_alpha_optimizer = optimizer_class(
                self.expert_log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.expert_log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch, batch_success=None):
        self._total_steps += 1

        # classic obs
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        # retrieve local experts information
        lambda_s_current = batch['use_local_current']
        lambda_s_next = batch['use_local_next']
        expert_actions = batch['expert_actions']
        next_expert_actions = batch['next_expert_actions']

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

        """ Parametrized noise loss"""
        parametrized_perturbation_actions, expert_log_pi = self.parametrized_perturbation(observations, expert_actions)

        if self.config.use_automatic_entropy_tuning_parametrized_perturbation:
            expert_alpha_loss = -(self.expert_log_alpha() * (expert_log_pi + self.config.target_entropy).detach()).mean()
            expert_alpha = self.expert_log_alpha().exp() * self.config.expert_alpha_multiplier
        else:
            expert_alpha_loss = observations.new_tensor(0.0)
            expert_alpha = observations.new_tensor(self.config.expert_alpha_multiplier)

        q_new_actions_perturbed = lambda_s_current * torch.min(
            self.qf1(observations, parametrized_perturbation_actions),
            self.qf2(observations, parametrized_perturbation_actions),
        )
        parametrized_perturbation_loss = (expert_alpha * expert_log_pi - q_new_actions_perturbed).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        with torch.no_grad():
            new_next_actions, next_log_pi = self.policy(next_observations)

            next_log_pi = (1 - lambda_s_next) * next_log_pi

            expert_target_q_values = torch.min(
                self.target_qf1(next_observations, next_expert_actions),
                self.target_qf2(next_observations, next_expert_actions),
            )
            classic_target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )
            target_q_values = lambda_s_next * expert_target_q_values + \
                              (1 - lambda_s_next) * classic_target_q_values

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

        if self.config.use_automatic_entropy_tuning_parametrized_perturbation:
            self.expert_alpha_optimizer.zero_grad()
            expert_alpha_loss.backward()
            self.expert_alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.parametrized_perturbation_optimizer.zero_grad()
        parametrized_perturbation_loss.backward()
        self.parametrized_perturbation_optimizer.step()

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
            parametrized_perturbation_loss=parametrized_perturbation_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            expert_alpha_loss=expert_alpha_loss.item(),
            expert_alpha=expert_alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        return metrics_to_return

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    def get_action(self,
                   env,
                   observation,
                   deterministic=False,
                   add_local_information=False):
        """
        In switched agent, the agent always picks the expert action if it is relevant.
        """

        action = self.sampler_policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
        if add_local_information:
            use_local = self.use_local.get_use_local(env,
                                                     observation)
            expert_action_init = self.local_expert.get_action(observation,
                                                              init_action=action,
                                                              env=env)
            expert_action = self.sampler_parametrized_perturbation(
                np.expand_dims(observation, 0), np.expand_dims(expert_action_init, 0),
                beta=self.beta, deterministic=deterministic
            )[0, :]
            if use_local:
                return expert_action, use_local, expert_action
            return action, use_local, expert_action
        return action

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        return modules

    @property
    def total_steps(self):
        return self._total_steps
