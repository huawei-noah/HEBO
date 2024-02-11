# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Dict, Union, List, Generator, Any
import numpy as np
import torch


class ReplayBuffer(object):
    """
    Replay buffer for storing and sampling transitions.

    Parameters:
    ----------
    max_size : int
        Maximum size of the replay buffer.
    data : dict, optional
        Initial data to populate the replay buffer.
    nb_local_experts : int, optional
        Number of local experts (default is 0).
    """

    def __init__(self, max_size: int, data: Optional[Dict[str, np.ndarray]] = None, nb_local_experts: Optional[int] = 0):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0
        self.nb_local_experts = nb_local_experts

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self) -> int:
        """
        Get the current size of the replay buffer.

        Returns:
        ----------
        int
            Current size of the replay buffer.
        """
        return self._size

    def _init_storage(self, observation_dim: int, action_dim: int) -> None:
        """
        Initialize the storage arrays.

        Parameters:
        ----------
        observation_dim : int
            Dimensionality of the observations.
        action_dim : int
            Dimensionality of the actions.

        Returns:
        ----------
        None
        """
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._use_local_current = np.zeros(self._max_size, dtype=np.float32)
        self._use_local_next = np.zeros(self._max_size, dtype=np.float32)
        self._expert_actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._next_expert_actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self,
                   observation: np.ndarray,
                   action: np.ndarray,
                   reward: float,
                   next_observation: np.ndarray,
                   done: bool,
                   use_local_current: float,
                   use_local_next: float,
                   expert_actions: np.ndarray,
                   next_expert_actions: np.ndarray):
        """
        Add a single transition to the replay buffer.

        Parameters:
        ----------
        observation : np.ndarray
            Observation array.
        action : np.ndarray
            Action array.
        reward : float
            Reward value.
        next_observation : np.ndarray
            Next observation array.
        done : bool
            Whether the episode is done.
        use_local_current : float
            Confidence function for local expert for the current action.
        use_local_next : float
            Confidence function for local expert for the next action.
        expert_actions : np.ndarray
            Expert actions array.
        next_expert_actions : np.ndarray
            Next expert actions array.

        Returns:
        ----------
        None
        """
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        # use locals
        self._use_local_current[self._next_idx] = float(use_local_current)
        self._use_local_next[self._next_idx] = float(use_local_next)

        # actions
        self._expert_actions[self._next_idx] = np.array(expert_actions, dtype=np.float32)
        self._next_expert_actions[self._next_idx] = np.array(next_expert_actions, dtype=np.float32)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                 next_observations: np.ndarray, dones: np.ndarray,
                 use_local_current: np.ndarray, use_local_next: np.ndarray,
                 expert_actions: np.ndarray, next_expert_actions: np.ndarray):
        """
        Add a trajectory to the replay buffer.

        Parameters:
        ----------
        observations : np.ndarray
            Array of observations.
        actions : np.ndarray
            Array of actions.
        rewards : np.ndarray
            Array of rewards.
        next_observations : np.ndarray
            Array of next observations.
        dones : np.ndarray
            Array of done flags.
        use_local_current : np.ndarray
            Array of flags for using local expert for the current action.
        use_local_next : np.ndarray
            Array of flags for using local expert for the next action.
        expert_actions : np.ndarray
            Array of expert actions.
        next_expert_actions : np.ndarray
            Array of next expert actions.

        Returns:
        ----------
        None
        """
        for o, a, r, no, d, u_c, u_n, ea, nea in zip(observations, actions, rewards, next_observations, dones,
                                                     use_local_current, use_local_next,
                                                     expert_actions, next_expert_actions):
            self.add_sample(o, a, r, no, d, u_c, u_n, ea, nea)

    def add_batch(self, batch: Dict[str, np.ndarray]):
        """
        Add a batch of data to the replay buffer.

        Parameters:
        ----------
        batch : Dict[str, np.ndarray]
            Dictionary containing arrays of observations, actions, rewards, next observations,
            done flags, floats for the confidence function of the local expert for the current action,
            floats for the confidence function of the local expert for the next action, expert actions,
            and next expert actions.

        Returns:
        ----------
        None
        """
        self.add_traj(
            batch['observations'],
            batch['actions'],
            batch['rewards'],
            batch['next_observations'],
            batch['dones'],
            batch['use_local_current'],
            batch['use_local_next'],
            batch['expert_actions'],
            batch['next_expert_actions'],
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of data from the replay buffer.

        Parameters:
        ----------
        batch_size : int
            The number of samples to be drawn.

        Returns:
        ----------
        Dict[str, np.ndarray]
            Dictionary containing arrays of observations, actions, rewards, next observations,
            done flags, flags for using local expert for the current action, flags for using local
            expert for the next action, expert actions, and next expert actions.
        """
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select samples from the replay buffer based on the given indices.

        Parameters:
        ----------
        indices : np.ndarray
            Array of indices to select samples from the replay buffer.

        Returns:
        ----------
        Dict[str, np.ndarray]
            Dictionary containing arrays of observations, actions, rewards, next observations,
            done flags, flags for using local expert for the current action, flags for using local
            expert for the next action, expert actions, and next expert actions.
        """
        use_local_current = self._use_local_current[indices, ...]
        use_local_next = self._use_local_next[indices, ...]
        expert_actions = self._expert_actions[indices, ...]
        next_expert_actions = self._next_expert_actions[indices, ...]

        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
            use_local_current=use_local_current,
            use_local_next=use_local_next,
            expert_actions=expert_actions,
            next_expert_actions=next_expert_actions
        )

    def generator(self, batch_size: int, n_batchs: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Generator function that yields batches of samples from the replay buffer.

        Parameters:
        ----------
        batch_size : int
            Size of each batch.
        n_batchs : int, optional
            Number of batches to generate (default is None for an infinite generator).

        Yields:
        ----------
        Dict[str, Any]
            Dictionary containing arrays of observations, actions, rewards, next observations,
            done flags, flags for using local expert for the current action, flags for using local
            expert for the next action, expert actions, and next expert actions.
        """
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self) -> int:
        """
        Property to get the total number of steps taken by the replay buffer.

        Returns:
        ----------
        int
            Total number of steps.
        """
        return self._total_steps

    @property
    def data(self) -> Dict[str, Any]:
        """
        Property to get a dictionary containing arrays of observations, actions, rewards, next observations,
        done flags, confidence function for using local expert for the current action,
        confidence function for using local expert for the next action, expert actions, and next expert actions.

        Returns:
        ----------
        Dict[str, Any]
            Dictionary containing arrays of observations, actions, rewards, next observations,
            done flags, confidence function for using local expert for the current action,
            confidence function for using local expert for the next action, expert actions, and next expert actions.
        """
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...],
            use_local_current=self._use_local_current[:self._size, ...],
            use_local_next=self._use_local_next[:self._size, ...],
            expert_actions=self._expert_actions[:self._size, ...],
            next_expert_actions=self._next_expert_actions[:self._size, ...],
        )


def batch_to_torch(batch: Dict[str, Union[np.ndarray, Dict[int, np.ndarray]]], device: str) \
        -> Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]]:
    """
    Convert a batch from NumPy arrays to PyTorch tensors.

    Parameters:
    ----------
    batch : Dict[str, Union[np.ndarray, Dict[int, np.ndarray]]]
        Dictionary containing NumPy arrays or dictionaries of NumPy arrays.
    device : str
        The device to which the tensors should be moved.

    Returns:
    ----------
    Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]]
        Dictionary containing PyTorch tensors or dictionaries of PyTorch tensors.
    """
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True) if type(v) is np.ndarray
        else {nb: torch.from_numpy(v[nb]).to(device=device, non_blocking=True) for nb in range(len(v))}
        for k, v in batch.items()
    }


def subsample_batch(batch: Dict[str, np.ndarray], size: int) -> Dict[str, np.ndarray]:
    """
    Subsample a batch with the given size.

    Parameters:
    ----------
    batch : Dict[str, np.ndarray]
        Dictionary containing NumPy arrays.
    size : int
        The size of the subsampled batch.

    Returns:
    ----------
    Dict[str, np.ndarray]
        Subsampled batch.
    """
    indices = np.random.randint(batch['observations'].shape[0], size=size)

    return dict(
        observations=batch['observations'][indices, ...],
        actions=batch['actions'][indices, ...],
        rewards=batch['rewards'][indices, ...],
        next_observations=batch['next_observations'][indices, ...],
        dones=batch['dones'][indices, ...],
        use_local_current=batch['use_local_current'][indices, ...],
        use_local_next=batch['use_local_next'][indices, ...],
        expert_actions=batch['expert_actions'][indices, ...],
        next_expert_actions=batch['next_expert_actions'][indices, ...],
    )


def concatenate_batches(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Concatenate multiple batches into a single batch.

    Parameters:
    ----------
    batches : List[Dict[str, np.ndarray]]
        List of dictionaries, each containing NumPy arrays.

    Returns:
    ----------
    Dict[str, np.ndarray]
        Concatenated batch.
    """
    return dict(
        observations=np.concatenate([batch['observations'] for batch in batches], axis=0).astype(np.float32),
        actions=np.concatenate([batch['actions'] for batch in batches], axis=0).astype(np.float32),
        rewards=np.concatenate([batch['rewards'] for batch in batches], axis=0).astype(np.float32),
        next_observations=np.concatenate([batch['next_observations'] for batch in batches], axis=0).astype(np.float32),
        dones=np.concatenate([batch['dones'] for batch in batches], axis=0).astype(np.float32),
        use_locals_current=np.concatenate([batch['use_locals_current'] for batch in batches], axis=0).astype(np.float32),
        use_locals_next=np.concatenate([batch['use_locals_next'] for batch in batches], axis=0).astype(np.float32),
        expert_actions=np.concatenate([batch['expert_actions'] for batch in batches], axis=0).astype(np.float32),
        next_expert_actions=np.concatenate([batch['next_expert_actions'] for batch in batches], axis=0).astype(np.float32),
    )
