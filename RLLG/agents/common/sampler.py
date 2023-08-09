# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) 2020 Xinyang Geng.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import numpy as np


class StepSampler(object):

    def __init__(self,
                 env,
                 max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, agent, n_steps, deterministic=False, replay_buffer=None):
        # general observations
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        list_use_local_current = []
        list_use_local_next = []
        failures = []

        for n_ in range(n_steps):

            self._traj_steps += 1
            observation = self._current_observation

            # get action and local information
            if n_ == 0:
                action, use_local_current, expert_action = agent.get_action(self.env,
                                                                            observation,
                                                                            deterministic=deterministic,
                                                                            add_local_information=True)
            else:
                expert_action = next_expert_action.copy()
                use_local_current = use_local_next
                action = next_action.copy()

            # Apply next action and save transition
            next_observation, reward, done, info = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            if reward <= -500:
                failures.append(1)
            else:
                failures.append(0)

            # Choose action according to local policies to record for both obs and next_obs
            next_action, use_local_next, next_expert_action = agent.get_action(self.env,
                                                                               next_observation,
                                                                               deterministic=deterministic,
                                                                               add_local_information=True)

            # add local information
            list_use_local_current.append(use_local_current)
            list_use_local_next.append(use_local_next)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                    use_local_current,
                    use_local_next,
                    expert_action,
                    next_expert_action
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._current_observation = self.env.reset()
                self._traj_steps = 0

        metrics_to_return = dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            list_use_local_current=np.array(list_use_local_current, dtype=np.float32),
            list_use_local_next=np.array(list_use_local_next, dtype=np.float32),
            failures=np.array(failures, dtype=np.float32),
        )

        return metrics_to_return

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self,
                 env,
                 max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, agent, n_trajs, deterministic=False, replay_buffer=None, replay_buffer_success=None):

        trajs = []

        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            failures = []
            list_use_local_current = []
            list_use_local_next = []

            observation = self.env.reset()

            for n_ in range(self.max_traj_length):

                # get action and local information
                if n_ == 0:
                    action, use_local_current, expert_action = agent.get_action(self.env,
                                                                                observation,
                                                                                deterministic=deterministic,
                                                                                add_local_information=True)
                else:
                    expert_action = next_expert_action.copy()
                    use_local_current = use_local_next
                    action = next_action.copy()

                # Apply next action and save transition
                next_observation, reward, done, info = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)
                if reward <= -500:
                    failures.append(1)
                else:
                    failures.append(0)

                # Choose action according to local policies to record for both obs and next_obs
                next_action, use_local_next, next_expert_action = agent.get_action(self.env,
                                                                                   next_observation,
                                                                                   deterministic=deterministic,
                                                                                   add_local_information=True)

                # add local information
                list_use_local_current.append(use_local_current)
                list_use_local_next.append(use_local_next)

                observation = next_observation

                if done:
                    break

            metrics_to_return = dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                list_use_local_current=np.array(list_use_local_current, dtype=np.float32),
                list_use_local_next=np.array(list_use_local_next, dtype=np.float32),
                failures=np.array(failures, dtype=np.float32),
            )

            trajs.append(metrics_to_return)

        return trajs

    @property
    def env(self):
        return self._env
