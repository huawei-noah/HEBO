import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
import numpy as np

from .MAF import RealNVP
from utils.utils import soft_clamp

epsilon = 1e-6
NOISE_STD = 0.2

LOG_SIG_MIN = -4
LOG_SIG_MAX = 2


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, BatchLinear):
        for i in range(m.batch):
            torch.nn.init.xavier_uniform_(m.weight[:, :, i], gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BatchLinear(nn.Module):
    def __init__(self, batch: int, in_features: int, out_features: int, first=False) -> None:
        super(BatchLinear, self).__init__()
        self.batch = batch
        self.first = first

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(out_features, in_features, batch)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(out_features, batch)))

    def forward(self, input: Tensor) -> Tensor:
        if self.first:
            return torch.einsum('...j,kjb->...kb', input, self.weight) + self.bias
        else:
            return torch.einsum('...jb,kjb->...kb', input, self.weight) + self.bias


class DoubleQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DoubleQNetwork, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)
        return torch.cat([self.net1(xu), self.net2(xu)], -1)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space,
                 min_log_std=np.log(1e-2)):
        super(GaussianPolicy, self).__init__()
        self.min_log_std = torch.tensor(min_log_std, dtype=torch.float32)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = soft_clamp(log_std, _min=LOG_SIG_MIN, _max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, num=None, det=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if not det:
            if num is None:
                x_t = normal.rsample()
            else:
                x_t = normal.rsample([num])
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= self.action_scale.log() + 2. * (np.log(2.) - x_t - F.softplus(-2. * x_t))
            log_prob = log_prob.sum(-1)
        else:
            y_t = torch.tanh(mean)
            action = y_t * self.action_scale + self.action_bias
            log_prob = None
        return action, log_prob

    def tanh_normal_sample(self, action, std, num=None):
        n_action = torch.clip((action - self.action_bias) / self.action_scale, -1. + epsilon, 1. - epsilon)
        r_action = torch.atanh(n_action)
        normal = Normal(r_action, std)
        if num is None:
            r_sample = normal.rsample()
        else:
            r_sample = normal.rsample([num])
        n_sample = torch.tanh(r_sample)
        sample = n_sample * self.action_scale + self.action_bias
        log_prob = normal.log_prob(r_sample)
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - r_sample - F.softplus(-2. * r_sample))
        log_prob = log_prob.sum(-1)
        return sample, log_prob

    def cal_prob(self, action, state, beta=0.):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        n_action = torch.clip((action - self.action_bias) / self.action_scale, -1.+epsilon, 1.-epsilon)
        r_action = torch.atanh(n_action)
        log_prob = normal.log_prob(r_action)
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - r_action - F.softplus(-2. * r_action))
        weight = std.detach() ** (2 * beta)
        weight /= weight.mean(-2, keepdim=True)
        log_prob *= weight
        log_prob = log_prob.sum(-1)
        return log_prob

    def to(self, device):
        self.min_log_std = self.min_log_std.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class RealNvpPolicy(RealNVP):
    def __init__(self, state_dim, action_dim, action_space):
        super(RealNvpPolicy, self).__init__(n_blocks=4, input_size=action_dim, hidden_size=action_dim, n_hidden=1,
                                            cond_label_size=2*action_dim, batch_norm=False)
        self.preprocessor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*action_dim),
            nn.ReLU()
        )

        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def forward(self, x, y=None):
        return super(RealNvpPolicy, self).forward(x, self.preprocessor(y))

    def inverse(self, u, y=None):
        return super(RealNvpPolicy, self).inverse(u, self.preprocessor(y))

    def sample(self, state, num=None):
        if num is None:
            noise = self.base_dist.sample(state.shape[:-1])
            state_ = state
        else:
            noise = self.base_dist.sample([num, *state.shape[:-1]])
            state_ = state.expand([num, *([-1] * state.ndim)])
        raw_action, sum_log_abs_det_jacobians = self.inverse(noise, state_)
        log_prob = self.base_dist.log_prob(noise) - sum_log_abs_det_jacobians

        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale + self.action_bias
        # Enforcing Action Bound
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - raw_action - F.softplus(-2. * raw_action))
        log_prob = log_prob.sum(-1)
        return action, log_prob

    def tanh_normal_sample(self, action, std, num=None):
        n_action = torch.clip((action - self.action_bias) / self.action_scale, -1. + epsilon, 1. - epsilon)
        r_action = torch.atanh(n_action)
        normal = Normal(r_action, std)
        if num is None:
            r_sample = normal.rsample()
        else:
            r_sample = normal.rsample([num])
        n_sample = torch.tanh((r_sample))
        sample = n_sample * self.action_scale + self.action_bias
        log_prob = normal.log_prob(r_sample)
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - r_sample - F.softplus(-2. * r_sample))
        log_prob = log_prob.sum(-1)
        return sample, log_prob

    def cal_prob(self, action, state, beta=0.):
        tanh_action = torch.clip((action - self.action_bias) / self.action_scale, -1. + epsilon, 1. - epsilon)
        raw_action = torch.atanh(tanh_action)
        state = state.expand([*action.shape[:(raw_action.ndim-state.ndim)], *([-1] * state.ndim)])
        log_prob = self.log_prob(raw_action, state)
        log_prob -= (self.action_scale.log() + 2. * (np.log(2.) - raw_action - F.softplus(-2. * raw_action))).sum(-1)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(RealNvpPolicy, self).to(device)
