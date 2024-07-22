#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 2021.11.10-Add A2C agent
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

from typing import Tuple, Optional

import utils.utils_misc
from DRiLLS.utils import get_model_path, get_playground_dir
from utils.utils_misc import log
import time
import os

import numpy as np
import tensorflow as tf

from .fpga_session import FPGASession as FPGAGame
from .models.agent import Agent
from .scl_session import SCLSession as SCLGame


class Normalizer:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.n = tf.zeros(num_inputs)
        self.mean = tf.zeros(num_inputs)
        self.mean_diff = tf.zeros(num_inputs)
        self.var = tf.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = tf.identity(self.mean)
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = tf.clip_by_value(self.mean_diff / self.n, clip_value_min=1e-2, clip_value_max=1000000000)

    def normalize(self, inputs):
        obs_std = tf.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def reset(self):
        self.n = tf.zeros(self.num_inputs)
        self.mean = tf.zeros(self.num_inputs)
        self.mean_diff = tf.zeros(self.num_inputs)
        self.var = tf.zeros(self.num_inputs)


class A2C(Agent):

    def __init__(self, options, design: str, design_file: str, max_iteration: int, load_model_from=None, fpga_mapping=False):
        """

        Args:
            options:
            design:
            load_model:
            fpga_mapping:
        """
        self.design = design
        self.design_file = design_file
        self.playground_dir = get_playground_dir(
            design=self.design,
            learner_id=self.model_id,
        )
        if fpga_mapping:
            self.game = FPGAGame(options)
        else:
            self.game = SCLGame(params=options,
                                design_file=self.design_file,
                                playground_dir=self.playground_dir, max_iteration=max_iteration)

        self.num_actions = self.game.action_space_length
        self.state_size = self.game.observation_space_size
        self.normalizer = Normalizer(self.state_size)

        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])

        # Define any additional placeholders needed for training your agent here:
        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ])

        self.state_value = self.critic()
        self.actor_probs = self.actor()
        self.loss_val = self.loss()
        self.train_op = self.optimizer()
        self.session = tf.Session()

        # model saving/restoring
        self.saver = tf.train.Saver()

        if load_model_from is not None:
            self.saver.restore(self.session, load_model_from)
            log("Model restored.")
        else:
            self.session.run(tf.global_variables_initializer())

        self.gamma = 0.99
        self.learning_rate = 0.01

    def optimizer(self):
        """
        :return: Optimizer for your loss function
        """
        return tf.train.AdamOptimizer(0.01).minimize(self.loss_val)

    def critic(self):
        """
        Calculates the estimated value for every state in self.state_input. The critic should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states] representing the estimated value of each state in the trajectory.
        """
        c_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                  num_outputs=10,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        c_fc2 = tf.contrib.layers.fully_connected(inputs=c_fc1,
                                                  num_outputs=1,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        return c_fc2

    def actor(self):
        """
        Calculates the action probabilities for every state in self.state_input. The actor should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states, num_actions] representing the probability distribution
            over actions that is generated by your actor.
        """
        a_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                  num_outputs=20,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        a_fc2 = tf.contrib.layers.fully_connected(inputs=a_fc1,
                                                  num_outputs=20,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        a_fc3 = tf.contrib.layers.fully_connected(inputs=a_fc2,
                                                  num_outputs=self.num_actions,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

        return tf.nn.softmax(a_fc3)

    def loss(self):
        """
        :return: A scalar tensor representing the combined actor and critic loss.
        """
        # critic loss
        advantage = self.discounted_episode_rewards_ - self.state_value
        critic_loss = tf.reduce_sum(tf.square(advantage))

        # actor loss        
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=utils.utils_misc.log(self.actor_probs),
                                                                  labels=self.actions)
        actor_loss = tf.reduce_sum(neg_log_prob * advantage)

        # policy_gradient_loss
        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actor_probs,
        #                                                           labels=self.actions)
        # return tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_)

        return critic_loss + actor_loss

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path()), exist_ok=True)
        save_path = self.saver.save(self.session, self.model_path())
        log("Model saved in path: %s" % str(save_path))

    def train(self, n_episodes: int):
        log(f"Starting to train the agent on {self.design}")
        print(f"Saving results in:\n\t{self.game.playground_dir}")

        all_rewards = []
        initial_delay, initial_area = self.get_design_prop()
        log(f"Initial delay: {initial_delay}")
        log(f"Initial area: {initial_area}")
        training_start_time = time.time()
        for i in range(n_episodes):
            log('Episode: ' + str(i + 1))
            start = time.time()
            total_reward = self.train_episode()
            end = time.time()
            all_rewards.append(total_reward)
            log('Episode: ' + str(i + 1) + ' - done with total reward = ' + str(total_reward))
            log('Episode ' + str(i + 1) + ' Run Time ~ ' + str((start - end) / 60) + ' minutes.')
            print('')
        training_end_time = time.time()
        log('Total Training Run Time ~ ' + str((training_end_time - training_start_time) / 60) + ' minutes.')

    def train_episode(self):
        """
        train_episode will be called several times by the drills.py to train the agent. In this method,
        we run the agent for a single episode, then use that data to train the agent.
        """
        state = self.game.reset()
        self.normalizer.reset()
        self.normalizer.observe(state)
        state = self.normalizer.normalize(state).eval(session=self.session)
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []

        while not done:
            log('  iteration: ' + str(self.game.iteration))
            action_probability_distribution = self.session.run(self.actor_probs,
                                                               feed_dict={self.state_input: state.reshape(
                                                                   [1, self.state_size])})
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())
            new_state, reward, done, _ = self.game.step(action)

            # append this step
            episode_states.append(state)
            action_ = np.zeros(self.num_actions)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)

            state = new_state
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state).eval(session=self.session)

        # Now that we have run the episode, we use this data to train the agent
        start = time.time()
        discounted_episode_rewards = self.discount_and_normalize_rewards(episode_rewards)

        _ = self.session.run(self.train_op, feed_dict={self.state_input: np.array(episode_states),
                                                       self.actions: np.array(episode_actions),
                                                       self.discounted_episode_rewards_: discounted_episode_rewards})
        end = time.time()
        log('Episode Agent Training Time ~ ' + str((start - end) / 60) + ' minutes.')
        log(f"Area: {self.game.area} | Delay: {self.game.delay} ({'un' if not self.game.constr_met() else ''}met)")
        self.save_model()

        return np.sum(episode_rewards)

    def discount_and_normalize_rewards(self, episode_rewards):
        """
        used internally to calculate the discounted episode rewards
        """
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)

        discounted_episode_rewards = (discounted_episode_rewards - mean) / std

        return discounted_episode_rewards

    def run_episode(self) -> Tuple[float, float, bool]:
        """
        Run one episode to test agent's policy
        """
        state = self.game.reset()
        self.normalizer.reset()
        self.normalizer.observe(state)
        state = self.normalizer.normalize(state).eval(session=self.session)
        done = False

        while not done:
            log('  iteration: ' + str(self.game.iteration))
            action_probability_distribution = self.session.run(self.actor_probs,
                                                               feed_dict={self.state_input: state.reshape(
                                                                   [1, self.state_size])})
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())
            new_state, _, done, _ = self.game.step(action)

            state = new_state
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state).eval(session=self.session)

        log(f"Area: {self.game.area} | Delay: {self.game.delay} ({'un' if not self.game.constr_met() else ''}met)")
        return self.game.area, self.game.delay, self.game.constr_met()

    def get_design_prop(self, time_constr: Optional[float] = None):
        """ Compute and return delay and area associated to a specific design without changing internal states of
                the game """
        return self.game.get_design_prop_(time_constr=time_constr)

    @property
    def model_id(self) -> str:
        return 'drills-tf-v0'

    def model_path(self) -> str:
        return get_model_path(
            design=self.design,
            learner_id=self.model_id,
        )
