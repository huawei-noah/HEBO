# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256',
                 activation="relu", return_last_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.activation = activation
        self.return_last_layer = return_last_layer

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            if self.activation == 'relu':
                modules.append(nn.ReLU())
            elif self.activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NotImplementedError(f'activation is {self.activation}')
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)

        if self.return_last_layer:
            self.network_but_last = nn.Sequential(*modules)
            self.last_fc = last_fc
        else:
            modules.append(last_fc)
            self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        if self.return_last_layer:
            last_layer = self.network_but_last(input_tensor)
            return self.last_fc(last_layer), last_layer.clone()
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0, no_tanh=False,
                 activation='relu'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch,
            activation=activation
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)



class ParametrizedPerturbationTanhGaussianPolicy(nn.Module):

    def __init__(self,
                 observation_dim,
                 action_dim,
                 arch='256-256',
                 log_std_multiplier=1.0,
                 log_std_offset=-1.0,
                 no_tanh=False,
                 activation='relu',
                 phi=0.5):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.no_tanh = no_tanh
        self.phi = phi

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch,
            activation=activation
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions, expert_actions):

        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()

        # get reversed actions to get the log prob of the expert parametrized policy
        phi_actions = (actions - expert_actions) / self.phi

        return self.tanh_gaussian.log_prob(mean, log_std, phi_actions)

    def forward(self, observations, expert_actions, beta=0., deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return (expert_actions + self.phi * (1 - beta) * actions).clamp(-0.999, 0.999), log_probs


class SamplerPolicy(object):

    def __init__(self, policy, device, from_ext=False):
        self.policy = policy
        self.device = device
        self.from_ext = from_ext

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return np.clip(actions, a_min=-0.999, a_max=0.999)


class ExpertSamplerPolicy(object):

    def __init__(self, policy, device, from_ext=False):
        self.policy = policy
        self.device = device
        self.from_ext = from_ext

    def __call__(self, observations, expert_actions, beta=1., deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            expert_actions = torch.tensor(
                expert_actions, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, expert_actions,
                                     beta=beta,
                                     deterministic=deterministic)
            actions = actions.cpu().numpy()
        return np.clip(actions, a_min=-0.999, a_max=0.999)


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 activation='relu', return_last_layer=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.return_last_layer = return_last_layer
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, activation=activation,
            return_last_layer=return_last_layer
        )

    def forward(self, observations, actions):
        if actions.ndim == 3 and observations.ndim == 2:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        if self.return_last_layer:
            output, last_layer = self.network(input_tensor)
            return torch.squeeze(output, dim=-1), last_layer
        output = self.network(input_tensor)
        return torch.squeeze(output, dim=-1)


class TD3Policy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256'):
        super(TD3Policy, self).__init__()
        self.arch = arch

        self.base_network = FullyConnectedNetwork(
            observation_dim, action_dim, arch
        )


    def forward(self, observation, deterministic=False):
        """
        Added the deterministic argument to be consitent with the code.
        """
        a_init = self.base_network(observation)
        return torch.tanh(a_init), None


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant
