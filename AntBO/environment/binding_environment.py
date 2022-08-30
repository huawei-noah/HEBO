import csv
import os
import time

import gym
import numpy as np
from gym import spaces


class BindingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, env_name, config, evaluator, rank=0, positive_reward=False):
        super(BindingEnvironment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.env_name = env_name

        self.n_sequences = config["sequence_length"]
        self.n_dimensions = config["dimensions"]
        self.model_tag = config["model_tag"]
        self.seed = config["seed"]
        self.algorithm = config["algorithm"]
        self.dir = config["directory"]
        self.positive_reward = positive_reward

        self.folder_name = "antigen_{}_kernel_{}_seed_{}_cdr_constraint_True_seqlen_11".format(self.model_tag,
                                                                                               self.algorithm,
                                                                                               str(seed))
        self.evaluator = evaluator

        if (self.env_name == "seq2seq"):
            self.action_space = spaces.Discrete(self.n_dimensions)
            self.observation_space = spaces.MultiBinary(self.n_sequences * (self.n_dimensions + 1))
            self.past_obs = np.zeros((self.n_sequences * (self.n_dimensions + 1),))
            self.maximum_steps = 11
            self.current_step = 0
            self.rank = str(rank)
            self.file = "results_cpu_{}.csv".format(self.rank)
            csv_columns = ["Index", "LastValue", "Time", "LastProtein"]
        elif (self.env_name == "one_shot"):
            self.n_antibody = config["n_antibody"]
            self.action_space = spaces.Tuple((spaces.Discrete(self.n_sequences), spaces.Discrete(self.n_dimensions)))
            self.observation_space = spaces.Tuple(
                (spaces.Discrete(self.n_sequences), spaces.Discrete(self.n_dimensions + 1)))
            self.file = "results.csv"
            self.best_value = 0
            self.best_protein = None
            csv_columns = ["Index", "LastValue", "BestValue", "Time", "LastProtein", "BestProtein"]
        else:
            raise NotImplementedError("{} is not an environment.".format(self.env_name))

        self.recorded_energy = {}
        self.start_time = time.time()
        self.index = 1
        self.save_path = "{}/{}/{}".format(self.dir, self.algorithm, self.folder_name)

        os.chdir(self.dir)
        if (os.path.exists(self.algorithm) == False):
            os.mkdir(self.algorithm)

        os.chdir(self.algorithm)

        if (os.path.exists(self.folder_name) == False):
            os.mkdir(self.folder_name)

        with open(self.file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)

    def step(self, action):
        assert self.action_space.contains(action)

        if (self.env_name == "seq2seq"):
            new_obs = np.zeros((self.n_sequences, (self.actions + 1))).astype(int)
            one_hot_action = np.zeros((self.actions + 1,)).astype(int)
            one_hot_action[action] = 1
            seq_act_past_obs = self.past_obs.reshape((self.n_sequences, (self.actions + 1)))
            new_obs[:-1] = seq_act_past_obs[1:]
            new_obs[-1] = one_hot_action
        else:
            new_obs = action

        if (self.current_step < self.maximum_steps - 1):
            reward = 0
            info = {}
            self.current_step += 1
            done = False
        else:
            # convert multibinary to discrete
            if (self.env_name == "seq2seq"):
                eval_obs = np.zeros((1, self.actions + 1)).astype(int)
                for i, seq in enumerate(new_obs):
                    eval_obs[0, i] = np.where(seq == 1)
            else:
                eval_obs = new_obs

            reward, seq = self.evaluator(eval_obs)
            if (self.env_name == "seq2seq"):
                reward = reward[0]
                seq = seq[0]

            info = {seq: reward}
            done = True

        if (self.env_name == "seq2seq"):
            new_obs = new_obs.flatten()
            self.past_obs = new_obs

        if (done):
            os.chdir(self.save_path)
            with open(self.file, 'a', newline='') as f:
                writer = csv.writer(f)

                if (self.env_name == "seq2seq"):
                    time_spent = time.time() - self.start_time
                    writer.writerow([self.index, reward, time_spent, seq])
                    self.index += 1
                else:
                    time_spent = (time.time() - self.start_time) / self.n_sequences
                    for binding, aa in zip(reward, seq):
                        if (binding < self.best_value):
                            self.best_value = binding
                            self.best_protein = aa

                        writer.writerow([self.index, binding, self.best_value, time_spent, aa, self.best_protein])
                        self.index += 1

            self.start_time = time.time()

            if (self.positive_reward == True):
                reward *= -1

        return new_obs, reward, done, info

    def reset(self):

        if (self.env_name == "seq2seq"):
            observation = np.zeros((self.n_sequences, (self.n_dimensions + 1))).astype(int).flatten()
            for seq in observation:
                seq[-1] = 1
        else:
            observation = np.zeros((self.n_sequences, self.n_dimensions)).astype(int) + (self.n_dimensions + 1)

        self.current_step = 0
        self.past_obs = observation

        return self.past_obs  # reward, done, info can't be included
