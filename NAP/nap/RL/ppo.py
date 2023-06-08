# Copyright (c) 2019
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# ppo.py
# Implementation of Proximal Policy Optimization as proposed in Schulman et al., https://arxiv.org/abs/1707.06347
# ******************************************************************

import os

from torch import nn
from torch.utils.tensorboard import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"  # on some machines this is needed to restrict torch to one core

import random
import torch
import torch.optim
import time
import numpy as np
import gym
import os
import pickle as pkl
import copy
import json
from datetime import datetime
import collections
from nap.RL.ppo_batchrecorder import BatchRecorder, Transition
from nap.policies.policies import iclr2020_NeuralAF


class PPO:
    def __init__(self, policy_fn, params, logpath, save_interval, verbose=False):
        self.params = params

        # set up the environment (only for reading out observation and action spaces)
        self.env = gym.make(self.params["env_id"])
        self.set_all_seeds()

        # logging
        self.logpath = logpath
        self.save_interval = save_interval
        self.verbose = verbose
        os.makedirs(logpath, exist_ok=True)
        # policies, optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pi = policy_fn(self.env.observation_space, self.env.action_space, deterministic=False)
        self.old_pi = policy_fn(self.env.observation_space, self.env.action_space, deterministic=False)
        if isinstance(self.pi, nn.Module):
            self.pi.to(self.device)
            self.old_pi.to(self.device)

            self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.params["lr"])
            if "load" in self.params and self.params["load"]:
                with open(os.path.join(self.params["load_path"], "weights_" + str(self.params["param_iter"])), "rb") as f:
                    print("loading weights_{}".format(self.params["param_iter"]))
                    self.pi.load_state_dict(torch.load(f,map_location="cpu"))
                    self.old_pi.load_state_dict(self.pi.state_dict())

        # set up the batch recorder
        self.batch_recorder = BatchRecorder(size=self.params["batch_size"],
                                            env_id=self.params["env_id"],
                                            env_seeds=self.params["env_seeds"],
                                            policy_fn=policy_fn,
                                            n_workers=self.params["n_workers"])

        self.stats = dict()
        self.stats["n_timesteps"] = 0
        self.stats["n_optsteps"] = 0
        self.stats["n_iters"] = 0
        self.stats["t_train"] = 0
        self.stats["avg_step_rews"] = np.array([])
        self.stats["avg_init_rews"] = np.array([])
        self.stats["avg_term_rews"] = np.array([])
        self.stats["avg_ep_rews"] = np.array([])
        self.writer = SummaryWriter(log_dir=os.path.join(self.logpath, f"tb0"))

        self.t_batch = None

        self.rew_buffer = collections.deque(maxlen=50)

        self.write_overview_logfile()

    def set_all_seeds(self):
        np.random.seed(self.params["seed"])
        random.seed(self.params["seed"])
        torch.manual_seed(self.params["seed"])
        torch.cuda.manual_seed_all(self.params["seed"])

    def write_overview_logfile(self):
        s = ""
        s += "********* OVERVIEW OF RUN *********\n"
        s += "Logpath        : {}\n".format(self.logpath)
        s += "Logfile created: {}\n".format(datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
        s += "Environment-ID:  {}\n".format(self.params["env_id"])
        s += "Environment-kwargs:\n"
        s += json.dumps(self.env.unwrapped.kwargs, indent=2)
        s += "\n"
        s += "PPO-parameters:\n"
        s += json.dumps(self.params, indent=2)
        s += "\n"
        s += "Batchrecorder:\n"
        s += json.dumps(self.batch_recorder.overview_dict(), indent=2)
        fname = "000_overview.txt"
        with open(os.path.join(self.logpath, fname), "w") as f:
            print(s, file=f)
        if not self.verbose:
            print(s)

    def optimize_on_batch(self):
        now = time.time()

        self.iter_log_str += " OPTIMIZING...\n"
        self.old_pi.load_state_dict(self.pi.state_dict())

        self.stats['loss_avg'] = []
        self.stats['loss_ppo_avg'] = []
        self.stats['loss_ent_avg'] = []
        self.stats['loss_value_avg'] = []

        for ep in range(self.params["n_epochs"]):
            loss_ppo_ep = 0
            loss_value_ep = 0
            loss_ent_ep = 0
            loss_ep = 0

            n_minibatches = 0
            for minibatch in self.batch_recorder.iterate(self.params["minibatch_size"], shuffle=True):
                transitions = Transition(*zip(*minibatch))
                states = torch.from_numpy(np.stack(transitions.state).astype(np.float32)).to(self.device)
                actions = torch.from_numpy(np.stack(transitions.action).astype(np.float32)).to(self.device)
                tdlamrets = torch.from_numpy(np.array(transitions.tdlamret).astype(np.float32)).to(self.device)

                if self.params["loss_type"] == "GAElam":
                    advs = torch.from_numpy(np.array(transitions.adv).astype(np.float32)).to(self.device)
                else:
                    advs = torch.from_numpy(np.array(transitions.tdlamret).astype(np.float32)).to(self.device)

                # normalize advantages
                if self.params["normalize_advs"]:
                    advs_std = torch.std(advs, unbiased=False)
                    if not advs_std == 0 and not torch.isnan(advs_std):
                        advs = (advs - torch.mean(advs)) / advs_std

                # compute values and entropies at current theta, and logprobs at current and old theta
                with torch.no_grad():
                    _, logprobs_old, _ = self.old_pi.predict_vals_logps_ents(states=states, actions=actions)
                vpreds, logprobs, entropies = self.pi.predict_vals_logps_ents(states=states, actions=actions)
                assert logprobs_old.dim() == vpreds.dim() == logprobs.dim() == entropies.dim() == 1

                # ppo-loss
                ratios = torch.exp(logprobs - logprobs_old)
                clipped_ratios = ratios.clamp(1 - self.params["epsilon"], 1 + self.params["epsilon"])
                advs = advs.squeeze()
                loss_cpi = ratios * advs
                assert loss_cpi.dim() == 1
                loss_clipped = clipped_ratios * advs
                assert loss_clipped.dim() == 1
                loss_ppo = -torch.mean(torch.min(loss_cpi, loss_clipped))

                # value-function loss
                loss_value = torch.mean((vpreds - tdlamrets) ** 2)

                # entropy loss
                loss_ent = -torch.mean(entropies)

                loss = loss_ppo + self.params["value_coeff"] * loss_value + self.params["ent_coeff"] * loss_ent

                with torch.no_grad():
                    loss_ppo_ep += loss_ppo
                    loss_value_ep += loss_value
                    loss_ent_ep += loss_ent
                    loss_ep += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.stats["n_optsteps"] += 1

                n_minibatches += 1

            loss_ppo_avg = loss_ppo_ep / n_minibatches
            loss_value_avg = loss_value_ep / n_minibatches
            loss_ent_avg = loss_ent_ep / n_minibatches
            loss_avg = loss_ep / n_minibatches
            self.iter_log_str += "   loss_ppo = {: .4g}, loss_value = {: .4g}, loss_ent = {: .4g}, loss = {: .4g}\n".format(
                loss_ppo_avg, loss_value_avg, loss_ent_avg, loss_avg)

            self.stats['loss_avg'].append(loss_avg)
            self.stats['loss_ppo_avg'].append(loss_ppo_avg)
            self.stats['loss_ent_avg'].append(loss_ent_avg)
            self.stats['loss_value_avg'].append(loss_value_avg)

        t_optim = time.time() - now
        self.iter_log_str += "  Took {:.2f}s".format(t_optim)

        return t_optim

    def train(self):
        while self.stats["n_timesteps"] < self.params["max_steps"]:
            self.iter_log_str = ""

            t_weights = self.batch_recorder.set_worker_weights(copy.deepcopy(self.pi))
            self.stats["t_train"] += t_weights

            t_batch = self.batch_recorder.record_batch(gamma=self.params["gamma"],
                                                       lam=self.params["lambda"])
            self.stats["t_train"] += t_batch
            batch_stats = self.batch_recorder.get_batch_stats()
            self.stats["n_timesteps"] += len(self.batch_recorder)
            self.stats["batch_stats"] = batch_stats
            self.stats["avg_step_rews"] = np.append(self.stats["avg_step_rews"], batch_stats["avg_step_reward"])
            self.stats["avg_init_rews"] = np.append(self.stats["avg_init_rews"], batch_stats["avg_initial_reward"])
            self.stats["avg_term_rews"] = np.append(self.stats["avg_term_rews"], batch_stats["avg_terminal_reward"])
            self.stats["avg_ep_rews"] = np.append(self.stats["avg_ep_rews"], batch_stats["avg_ep_reward"])
            self.stats["perc"] = 100 * self.stats["n_timesteps"] / self.params["max_steps"]
            batch_stats["t_batch"] = t_batch
            batch_stats["sps"] = batch_stats["size"] / batch_stats["t_batch"]
            self.add_iter_log_str(batch_stats=batch_stats)

            if self.stats["n_iters"] % self.save_interval == 0 or \
                    self.stats["n_iters"] == 0 or \
                    self.stats["n_timesteps"] >= self.params["max_steps"]:
                self.store_weights()

            t_optim = self.optimize_on_batch()
            self.stats["t_train"] += t_optim
            self.stats["t_optim"] = t_optim
            self.store_log()

            self.stats["n_iters"] += 1
        self.batch_recorder.cleanup()

    def test(self):
        # precompute seeds and dataset indexes to dispatch to workers
        self.set_all_seeds()
        number_of_task = int(self.params["max_steps"] / gym.spec(self.params['env_id']).max_episode_steps)
        number_of_datasets = len(gym.spec(self.params['env_id']).kwargs['f_opts']['data'])
        assert self.params["max_steps"] % gym.spec(self.params['env_id']).max_episode_steps == 0
        seed_list = np.random.randint(low=0, high=2**32 - 1, size=number_of_task)
        dataset_index_list = [i % number_of_datasets for i in range(number_of_task)]
        self.batch_recorder.dispatch_seeds(seed_list, dataset_index_list)
        assert len(seed_list) == len(dataset_index_list)

        if isinstance(self.pi, nn.Module):
            self.batch_recorder.set_worker_weights(copy.deepcopy(self.pi))

        self.teststats = {"avg_ep_reward": [], "regret": []}

        while self.stats["n_timesteps"] < self.params["max_steps"]:
            self.iter_log_str = ""
            self.stats["t_train"] += 0

            t_batch = self.batch_recorder.record_batch(gamma=self.params["gamma"], lam=self.params["lambda"])
            self.stats["t_train"] += t_batch
            batch_stats = self.batch_recorder.get_batch_stats()

            self.stats["n_timesteps"] += len(self.batch_recorder)
            self.stats["batch_stats"] = batch_stats

            self.teststats["avg_ep_reward"].append(batch_stats["avg_ep_reward"])
            self.teststats["regret"].extend(batch_stats["regret"])

            self.stats["avg_step_rews"] = np.append(self.stats["avg_step_rews"], batch_stats["avg_step_reward"])
            self.stats["avg_init_rews"] = np.append(self.stats["avg_init_rews"], batch_stats["avg_initial_reward"])
            self.stats["avg_term_rews"] = np.append(self.stats["avg_term_rews"], batch_stats["avg_terminal_reward"])
            self.stats["avg_ep_rews"] = np.append(self.stats["avg_ep_rews"], batch_stats["avg_ep_reward"])
            self.stats["perc"] = 100 * self.stats["n_timesteps"] / self.params["max_steps"]
            batch_stats["t_batch"] = t_batch
            batch_stats["sps"] = batch_stats["size"] / batch_stats["t_batch"]
            self.add_iter_log_str(batch_stats=batch_stats)

            self.store_log()
            self.stats["t_optim"] = 0
            self.stats["n_iters"] += 1
            self.store_bo_trajectories()

        assert self.batch_recorder.all_done()
        self.gen_bo_plot()
        self.batch_recorder.cleanup()

    def store_bo_trajectories(self):
        X, Y, R, T = zip(*self.batch_recorder.memory_xy)
        states_x = torch.from_numpy(np.stack(X))
        states_y = torch.from_numpy(np.stack(Y))
        states_r = torch.from_numpy(np.stack(R))
        states_t = torch.from_numpy(np.stack(T))

        if os.path.exists(f"{self.logpath}/x.bin"):
            prev_x = torch.load(f"{self.logpath}/x.bin")
            prev_y = torch.load(f"{self.logpath}/y.bin")
            prev_r = torch.load(f"{self.logpath}/r.bin")
            prev_t = torch.load(f"{self.logpath}/t.bin")
            states_x = torch.cat([prev_x, states_x], 0)
            states_y = torch.cat([prev_y, states_y], 0)
            states_r = torch.cat([prev_r, states_r], 0)
            states_t = torch.cat([prev_t, states_t], 0)

        torch.save(states_x, f"{self.logpath}/x.bin")
        torch.save(states_y, f"{self.logpath}/y.bin")
        torch.save(states_r, f"{self.logpath}/r.bin")
        torch.save(states_t, f"{self.logpath}/t.bin")

    def gen_bo_plot(self):
        r = torch.load(f"{self.logpath}/r.bin")
        try:
            import plotext as plt
            plt.plot(r.mean(0))
            plt.plot(r.mean(0) - r.std(0).squeeze(-1))
            plt.plot(r.mean(0) + r.std(0).squeeze(-1))
            plt.show()
        except:
            pass

    def store_log(self):
        with open(os.path.join(self.logpath, "log"), "a") as f:
            print(self.iter_log_str, file=f)
        if not self.verbose:
            print(self.iter_log_str)
        # with open(os.path.join(self.logpath, "stats_" + str(self.stats["n_iters"])), "wb") as f:
        #     pkl.dump(self.stats, f)
        # with open(os.path.join(self.logpath, "params_" + str(self.stats["n_iters"])), "wb") as f:
        #     pkl.dump(self.params, f)

        for k in ['n_timesteps', 'n_optsteps', 't_train', 'perc']:
            self.writer.add_scalar(k, self.stats[k], self.stats["n_iters"])
        for k in ['avg_step_reward', 'avg_initial_reward', 'avg_terminal_reward', 'avg_ep_reward', 'avg_ep_len',
                  't_batch', 'sps', 'regret_avg']:
            self.writer.add_scalar(k, self.stats['batch_stats'][k], self.stats["n_iters"])
        if 'loss_avg' in self.stats:
            for k in ['loss_avg', 'loss_ppo_avg', 'loss_ent_avg', 'loss_value_avg']:
                for val in self.stats[k]:
                    self.writer.add_scalar(k, val, self.stats["n_iters"])
        self.writer.flush()

    def store_weights(self):
        with open(os.path.join(self.logpath, "weights_" + str(self.stats["n_iters"])), "wb") as f:
            torch.save(self.pi.state_dict(), f)

    def add_iter_log_str(self, batch_stats):
        self.iter_log_str += "\n******************** ITERATION {:2d} ********************\n".format(
            self.stats["n_iters"])
        self.iter_log_str += " RUN STATISTICS (BEFORE OPTIMIZATION):\n"
        self.iter_log_str += "   environment          = {}\n".format(self.params["env_id"])
        self.iter_log_str += "   n_timesteps  = {:d} ({:.2f}%)\n".format(self.stats["n_timesteps"], self.stats["perc"])
        self.iter_log_str += "   n_optsteps   = {:d}\n".format(self.stats["n_optsteps"])
        self.iter_log_str += "   t_total      = {:.2f}s\n".format(self.stats["t_train"])
        self.iter_log_str += " BATCH STATISTICS (BEFORE OPTIMIZATION):\n"
        self.iter_log_str += "   n_workers    = {:d}\n".format(self.batch_recorder.n_workers)
        self.iter_log_str += "   worker_seeds = {}\n".format(self.batch_recorder.env_seeds)
        self.iter_log_str += "   size         = {:d}\n".format(batch_stats["size"])
        self.iter_log_str += "    per_worker  = {:}\n".format(batch_stats["worker_sizes"])
        self.iter_log_str += "   avg_step_rew = {:.4g}\n".format(batch_stats["avg_step_reward"])
        self.iter_log_str += "   avg_init_rew = {:.4g}\n".format(batch_stats["avg_initial_reward"])
        self.iter_log_str += "   avg_term_rew = {:.4g}\n".format(batch_stats["avg_terminal_reward"])
        self.iter_log_str += "   avg_ep_rew   = {:.4g}\n".format(batch_stats["avg_ep_reward"])
        self.iter_log_str += "   regret       = {:.4g}\n".format(batch_stats["regret_avg"])
        self.iter_log_str += "   n_new        = {:d}\n".format(batch_stats["n_new"])
        self.iter_log_str += "    per_worker  = {:}\n".format(batch_stats["worker_n_news"])
        self.iter_log_str += "   avg_ep_len   = {:.2f}\n".format(batch_stats["avg_ep_len"])
        self.iter_log_str += "   t_batch      = {:.2f}s ({:.0f}sps)\n".format(batch_stats["t_batch"],
                                                                              batch_stats["sps"])
