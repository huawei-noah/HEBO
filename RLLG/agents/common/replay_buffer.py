# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, data=None, nb_local_experts=0):
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

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
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

    def add_sample(self, observation, action, reward, next_observation, done,
                   use_local_current, use_local_next, expert_actions, next_expert_actions):
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

    def add_traj(self, observations, actions, rewards, next_observations, dones,
                 use_local_current, use_local_next, expert_actions, next_expert_actions):
        for o, a, r, no, d, u_c, u_n, ea, nea in zip(observations, actions, rewards, next_observations, dones,
                                                     use_local_current, use_local_next,
                                                     expert_actions, next_expert_actions):
            self.add_sample(o, a, r, no, d, u_c, u_n, ea, nea)

    def add_batch(self, batch):
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

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        # select expert if any
        use_locals_current, use_locals_next = {}, {}
        expert_actions, next_expert_actions = {}, {}
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

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
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


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True) if type(v) is np.ndarray
        else {nb: torch.from_numpy(v[nb]).to(device=device, non_blocking=True) for nb in range(len(v))}
        for k, v in batch.items()
    }


def subsample_batch(batch, size):
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


def concatenate_batches(batches):

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
