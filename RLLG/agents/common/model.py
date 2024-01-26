# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    """
    Extend and repeat the tensor along the specified axis.

    Parameters:
    ----------
    tensor : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to extend and repeat.
    repeat : int
        Number of times to repeat the tensor.

    Returns:
    ----------
    torch.Tensor
        Extended and repeated tensor.
    """
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network: nn.Module, target_network: nn.Module, soft_target_update_rate: float) -> None:
    """
    Update the target network parameters using a soft update.

    Parameters:
    ----------
    network : nn.Module
        The source network.
    target_network : nn.Module
        The target network to be updated.
    soft_target_update_rate : float
        The soft update rate.

    Returns:
    ----------
    None
    """
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


class FullyConnectedNetwork(nn.Module):
    """
    Fully connected neural network module.

    Parameters:
    ----------
    input_dim : int
        Dimension of the input.
    output_dim : int
        Dimension of the output.
    arch : str, optional
        Architecture of the network (default is '256-256').
    activation : str, optional
        Activation function (default is 'relu').
    return_last_layer : bool, optional
        Whether to return only the last layer (default is False).
    """

    def __init__(self, input_dim: int, output_dim: int, arch: Optional[str] = '256-256',
                 activation: Optional[str] = "relu", return_last_layer: Optional[bool] = False):
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

    def forward(self, input_tensor: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns:
        ----------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            The output of the network, and optionally, the output of the last layer.
        """
        if self.return_last_layer:
            last_layer = self.network_but_last(input_tensor)
            return self.last_fc(last_layer), last_layer.clone()
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):
    """
    Tanh Gaussian distribution with reparametrized trick.

    Parameters:
    ----------
    log_std_min : Optional[float], optional
        Minimum value for the log standard deviation (default is -20.0).
    log_std_max : Optional[float], optional
        Maximum value for the log standard deviation (default is 2.0).
    no_tanh : Optional[bool], optional
        Whether to skip applying tanh to the sampled actions (default is False).
    """

    def __init__(self, log_std_min: Optional[float] = -20.0,
                 log_std_max: Optional[float] = 2.0,
                 no_tanh: Optional[bool] = False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of a sample under the distribution.

        Parameters:
        ----------
        mean : torch.Tensor
            Mean of the distribution.
        log_std : torch.Tensor
            Log standard deviation of the distribution.
        sample : torch.Tensor
            Sample to compute the log probability for.

        Returns:
        ----------
        torch.Tensor
            Log probability of the sample.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: Optional[bool] = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Generate a sample and compute the log probability.

        Parameters:
        ----------
        mean : torch.Tensor
            Mean of the distribution.
        log_std : torch.Tensor
            Log standard deviation of the distribution.
        deterministic : bool, optional
            Flag indicating whether to sample deterministically (default is False).

        Returns:
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            Generated action sample and its log probability.
        """
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
    """
    Policy module representing a Tanh Gaussian policy.

    Parameters:
    ----------
    observation_dim : int
        Dimensionality of the observation space.
    action_dim : int
        Dimensionality of the action space.
    arch : str, optional
        Architecture of the base network (default is '256-256').
    log_std_multiplier : float, optional
        Multiplier for the log standard deviation (default is 1.0).
    log_std_offset : float, optional
        Offset for the log standard deviation (default is -1.0).
    no_tanh : bool, optional
        Whether to skip applying tanh to the sampled actions (default is False).
    activation : str, optional
        Activation function used in the base network (default is 'relu').
    """

    def __init__(self, observation_dim: int, action_dim: int, arch: Optional[str] = '256-256',
                 log_std_multiplier: Optional[float] = 1.0, log_std_offset: Optional[float] = -1.0,
                 no_tanh: Optional[bool] = False, activation: Optional[str] = 'relu'):
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

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of a given set of actions.

        Parameters:
        ----------
        observations : torch.Tensor
            Observations to condition the policy on.
        actions : torch.Tensor
            Actions for which to compute the log probability.

        Returns:
        ----------
        torch.Tensor
            Log probability of the given actions.
        """
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations: torch.Tensor, deterministic: bool = False, repeat: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Generate a sample and compute the log probability.

        Parameters:
        ----------
        observations : torch.Tensor
            Observations to condition the policy on.
        deterministic : bool, optional
            Flag indicating whether to sample deterministically (default is False).
        repeat : Optional[int], optional
            Number of times to repeat the action sampling (default is None).

        Returns:
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            Generated action sample and its log probability.
        """
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)



class ParametrizedPerturbationTanhGaussianPolicy(nn.Module):
    """
    Policy module representing the parametrized perturbation Tanh Gaussian policy.

    Parameters:
    ----------
    observation_dim : int
        Dimensionality of the observation space.
    action_dim : int
        Dimensionality of the action space.
    arch : str, optional
        Architecture of the base network (default is '256-256').
    log_std_multiplier : float, optional
        Multiplier for the log standard deviation (default is 1.0).
    log_std_offset : float, optional
        Offset for the log standard deviation (default is -1.0).
    no_tanh : bool, optional
        Whether to skip applying tanh to the sampled actions (default is False).
    activation : str, optional
        Activation function used in the base network (default is 'relu').
    phi : float, optional
        Phi parameter for the perturbation (default is 0.5).
    """

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 arch: Optional[str] = '256-256',
                 log_std_multiplier: Optional[float] = 1.0,
                 log_std_offset: Optional[float] = -1.0,
                 no_tanh: Optional[bool] = False,
                 activation: Optional[str] = 'relu',
                 phi: Optional[float] = 0.5):
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

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of a given set of actions with respect to expert actions.

        Parameters:
        ----------
        observations : torch.Tensor
            Observations to condition the policy on.
        actions : torch.Tensor
            Actions for which to compute the log probability.
        expert_actions : torch.Tensor
            Expert actions to condition the policy on.

        Returns:
        ----------
        torch.Tensor
            Log probability of the given actions with respect to expert actions.
        """
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()

        # get reversed actions to get the log prob of the expert parametrized policy
        phi_actions = (actions - expert_actions) / self.phi

        return self.tanh_gaussian.log_prob(mean, log_std, phi_actions)

    def forward(self, observations: torch.Tensor, expert_actions: torch.Tensor, beta: float = 0.,
                deterministic: bool = False, repeat: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sample and compute the log probability with respect to expert actions.

        Parameters:
        ----------
        observations : torch.Tensor
            Observations to condition the policy on.
        expert_actions : torch.Tensor
            Expert actions to condition the policy on.
        beta : float, optional
            Beta parameter for the perturbation (default is 0.).
        deterministic : bool, optional
            Flag indicating whether to sample deterministically (default is False).
        repeat : Optional[int], optional
            Number of times to repeat the action sampling (default is None).

        Returns:
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            Generated action sample and its log probability.
        """
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return (expert_actions + self.phi * (1 - beta) * actions).clamp(-0.999, 0.999), log_probs


class SamplerPolicy(object):
    """
    Wrapper class for creating a callable policy for action sampling.

    Parameters:
    ----------
    policy : nn.Module
        Policy module used for action sampling.
    device : torch.device
        Device on which to perform the action sampling.
    from_ext : bool, optional
        Flag indicating whether the policy is from an external source (default is False).
    """

    def __init__(self, policy: nn.Module, device: torch.device, from_ext: bool = False):
        self.policy = policy
        self.device = device
        self.from_ext = from_ext

    def __call__(self, observations: Union[torch.Tensor, np.ndarray], deterministic: bool = False) -> np.ndarray:
        """
        Sample actions from the policy.

        Parameters:
        ----------
        observations : Union[torch.Tensor, np.ndarray]
            Observations to condition the policy on.
        deterministic : bool, optional
            Flag indicating whether to sample deterministically (default is False).

        Returns:
        ----------
        np.ndarray
            Sampled actions.
        """
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return np.clip(actions, a_min=-0.999, a_max=0.999)


class ExpertSamplerPolicy(object):
    """
    Wrapper class for creating a callable policy for expert action sampling.

    Parameters:
    ----------
    policy : nn.Module
        Policy module used for expert action sampling.
    device : torch.device
        Device on which to perform the expert action sampling.
    from_ext : bool, optional
        Flag indicating whether the policy is from an external source (default is False).
    """

    def __init__(self, policy: nn.Module, device: torch.device, from_ext: bool = False):
        self.policy = policy
        self.device = device
        self.from_ext = from_ext

    def __call__(self, observations: Union[torch.Tensor, np.ndarray],
                 expert_actions: Union[torch.Tensor, np.ndarray], beta: float = 1.,
                 deterministic: bool = False) -> np.ndarray:
        """
        Sample expert actions from the policy.

        Parameters:
        ----------
        observations : Union[torch.Tensor, np.ndarray]
            Observations to condition the expert policy on.
        expert_actions : Union[torch.Tensor, np.ndarray]
            Expert actions to condition the expert policy on.
        beta : float, optional
            Weighting factor for blending expert actions (default is 1.).
        deterministic : bool, optional
            Flag indicating whether to sample expert actions deterministically (default is False).

        Returns:
        ----------
        np.ndarray
            Sampled expert actions.
        """
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
    """
    Fully connected Q-function neural network.

    Parameters:
    ----------
    observation_dim : int
        Dimension of the observation space.
    action_dim : int
        Dimension of the action space.
    arch : str, optional
        Architecture configuration for the fully connected layers (default is '256-256').
    activation : str, optional
        Activation function to use in the hidden layers (default is 'relu').
    return_last_layer : bool, optional
        Whether to return the activations of the last hidden layer (default is False).
    """

    def __init__(self, observation_dim: int, action_dim: int, arch: Optional[str] = '256-256',
                 activation: Optional[str] = 'relu', return_last_layer: Optional[bool] = False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.return_last_layer = return_last_layer
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, activation=activation,
            return_last_layer=return_last_layer
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q-function.

        Parameters:
        ----------
        observations : torch.Tensor
            Input observations.
        actions : torch.Tensor
            Input actions.

        Returns:
        ----------
        torch.Tensor
            Q-values for the given observations and actions.
        """
        if actions.ndim == 3 and observations.ndim == 2:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        if self.return_last_layer:
            output, last_layer = self.network(input_tensor)
            return torch.squeeze(output, dim=-1), last_layer
        output = self.network(input_tensor)
        return torch.squeeze(output, dim=-1)


class TD3Policy(nn.Module):
    """
    Twin Delayed DDPG (TD3) policy network.

    Parameters:
    ----------
    observation_dim : int
        Dimension of the observation space.
    action_dim : int
        Dimension of the action space.
    arch : str, optional
        Architecture configuration for the fully connected layers (default is '256-256').
    """

    def __init__(self, observation_dim: int, action_dim: int, arch: str = '256-256'):
        super(TD3Policy, self).__init__()
        self.arch = arch

        self.base_network = FullyConnectedNetwork(
            observation_dim, action_dim, arch
        )

    def forward(self, observation: torch.Tensor, deterministic: Optional[bool] = False) -> Tuple[torch.Tensor, None]:
        """
        Forward pass of the TD3 policy network.

        Parameters:
        ----------
        observation : torch.Tensor
            Input observation.
        deterministic : bool, optional
            Whether to use deterministic policy (default is False). Added it for code consistency.

        Returns:
        ----------
        Tuple[torch.Tensor, None]
            Tuple containing the action tensor and None (no auxiliary information).
        """
        a_init = self.base_network(observation)
        return torch.tanh(a_init), None


class Scalar(nn.Module):
    """
    Scalar value represented as a learnable parameter.

    Parameters:
    ----------
    init_value : float
        Initial value for the scalar.
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        """
        Forward pass to retrieve the scalar value.

        Returns:
        ----------
        torch.Tensor
            Learnable scalar value.
        """
        return self.constant
