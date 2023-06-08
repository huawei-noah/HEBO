# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from nap.RL.ppo_batchrecorder_nap import BatchRecorderNAP, TransitionNAP

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
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class PPO_NAP:
    def __init__(self, policy_fn, params, logpath, save_interval, verbose=False, shift_gpu=False):
        self.params = params

        # set up the environment (only for reading out observation and action spaces)
        self.env = gym.make(self.params["env_id"])
        self.set_all_seeds()

        # logging
        self.logpath = logpath
        self.save_interval = save_interval
        self.verbose = verbose

        # policies, optimizer
        self.ddp = dist.is_available() and dist.is_initialized()
        if self.ddp:
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        if torch.cuda.device_count() > 0:
            self.device = self.rank % torch.cuda.device_count()
        else:
            self.device = 'cpu'

        self.pi = policy_fn(self.env.observation_space, self.env.action_space, deterministic=False, dataparallel=self.ddp)
        self.scheduler = None
        if isinstance(self.pi, nn.Module):
            self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.params["lr"])
            if params['decay_lr']:
                self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 20, params['max_steps'] // params['batch_size'])
            if "load" in self.params and self.params["load"]:
                self.load_model()

        # set up the batch recorder
        self.batch_recorder = BatchRecorderNAP(size=self.params["batch_size"],
                                               env_id=self.params["env_id"],
                                               env_seeds=self.params["env_seeds"],
                                               policy_fn=policy_fn,
                                               n_workers=self.params["n_workers"],
                                               shift_gpu=shift_gpu)

        self.reset_stats()
        self.t_batch = None

        self.rew_buffer = collections.deque(maxlen=50)

        if self.rank == 0:
            self.write_overview_logfile()

    def set_all_seeds(self):
        np.random.seed(self.params["seed"])
        random.seed(self.params["seed"])
        torch.manual_seed(self.params["seed"])
        torch.cuda.manual_seed_all(self.params["seed"])

    def reset_stats(self):
        self.stats = dict()
        self.stats["n_timesteps"] = 0
        self.stats["n_optsteps"] = 0
        self.stats["n_iters"] = 0
        self.stats["t_train"] = 0
        self.stats["avg_step_rews"] = np.array([])
        self.stats["avg_init_rews"] = np.array([])
        self.stats["avg_term_rews"] = np.array([])
        self.stats["avg_ep_rews"] = np.array([])

        os.makedirs(self.logpath, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.logpath, f"tb{self.rank}"))

    def load_model(self, with_optimizer=True):
        with open(os.path.join(self.params["load_path"], "weights_" + str(self.params["param_iter"])), "rb") as f:
            print("loading weights_{}".format(self.params["param_iter"]))
            sdict = torch.load(f, map_location="cpu")
            if not self.ddp:
                keys = list(sdict.keys())
                for key in keys:
                    if "module." in key:
                        sdict[key.replace('module.', '')] = sdict[key]
                        del sdict[key]
            else:
                keys = list(sdict.keys())
                already_ddp = all([".module." in key for key in keys])
                if not already_ddp:
                    for key in keys:
                        sdict[key.replace('policy_net.', 'policy_net.module.')] = sdict[key]
                        del sdict[key]

            try:
                self.pi.load_state_dict(sdict)
            except:
                print("Warning: fail to load model with strict=True")
                self.pi.load_state_dict(sdict, strict=False)

            if hasattr(self.pi.policy_net, 'buckets') or 'buckets.borders' in ''.join(sdict.keys()):
                if isinstance(self.params['policy_options']['arch_spec']['y_range'], tuple):
                    assert np.isclose(self.pi.policy_net.buckets.borders[0].detach().cpu().numpy(),
                                         self.params['policy_options']['arch_spec']['y_range'][0]), \
                        f"{self.pi.policy_net.buckets.borders[0]} vs " \
                        f"{self.params['policy_options']['arch_spec']['y_range'][0]}"
                    assert np.isclose(self.pi.policy_net.buckets.borders[-1].detach().cpu().numpy(),
                                         self.params['policy_options']['arch_spec']['y_range'][1]), \
                        f"{self.pi.policy_net.buckets.borders[1]} vs " \
                        f"{self.params['policy_options']['arch_spec']['y_range'][1]}"
                else:
                    assert np.isclose(self.pi.policy_net.buckets.borders[0].detach().cpu().numpy(),
                                         -self.params['policy_options']['arch_spec']['y_range']), \
                        f"{self.pi.policy_net.buckets.borders[0]} vs " \
                        f"{-self.params['policy_options']['arch_spec']['y_range']}"
                    assert np.isclose(self.pi.policy_net.buckets.borders[-1].detach().cpu().numpy(),
                                         self.params['policy_options']['arch_spec']['y_range']), \
                        f"{self.pi.policy_net.buckets.borders[1]} vs " \
                        f"{self.params['policy_options']['arch_spec']['y_range']}"

        if with_optimizer:
            if os.path.exists(os.path.join(self.params["load_path"], "optim_weights_" + str(self.params["param_iter"]))):
                with open(os.path.join(self.params["load_path"], "optim_weights_" + str(self.params["param_iter"])),
                          "rb") as f:
                    try:
                        self.optimizer.load_state_dict(torch.load(f, map_location="cpu"))
                    except:
                        print("Warning: Failed to load optimiser state")
            else:
                print("WARNING: optimizer weight not loaded!")

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

        total_gather_time = 0
        network_process_time = 0

        self.iter_log_str += " OPTIMIZING...\n"
        self.stats['loss_avg'] = []
        self.stats['loss_ppo_avg'] = []
        self.stats['loss_ent_avg'] = []
        self.stats['loss_value_avg'] = []
        self.stats['loss_ce_avg'] = []
        self.stats['ent_inside_avg'] = []

        for ep in range(self.params["n_epochs"]):
            loss_ppo_ep = 0
            loss_value_ep = 0
            loss_ent_ep = 0
            loss_ep = 0
            loss_ce_ep = 0
            ent_inside_ep = 0

            n_minibatches = 0
            s_start = time.time()
            for minibatch in self.batch_recorder.iterate(self.params["minibatch_size"], shuffle=True):
                transitions = TransitionNAP(*zip(*minibatch))
                states_x = torch.from_numpy(np.stack(transitions.state_x).astype(np.float32)).to(self.device)
                states_y = torch.from_numpy(np.stack(transitions.state_y).astype(np.float32)).to(self.device)
                y_true = torch.from_numpy(np.stack(transitions.y_true).astype(np.float32)).to(self.device)
                states_ix = torch.from_numpy(np.stack(transitions.state_x_indep).astype(np.float32)).to(self.device)
                states_mask = torch.from_numpy(np.stack(transitions.state_mask)).to(self.device)
                state_request = torch.from_numpy(np.stack(transitions.state_request)).to(self.device)
                states_mask_shift = torch.from_numpy(np.stack(transitions.state_mask_shift)).to(self.device)
                actions = torch.from_numpy(np.stack(transitions.action).astype(np.float32)).to(self.device)
                proba_mask = torch.from_numpy(np.stack(transitions.proba_mask).astype(bool)).to(self.device)
                tdlamrets = torch.from_numpy(np.array(transitions.tdlamret).astype(np.float32)).to(self.device)
                logprobs_old = torch.from_numpy(np.array(transitions.old_logp).astype(np.float32)).to(self.device)
                total_gather_time += time.time() - s_start

                if self.params["loss_type"] == "GAElam":
                    advs = torch.from_numpy(np.array(transitions.adv).astype(np.float32)).to(self.device)
                else:
                    advs = torch.from_numpy(np.array(transitions.tdlamret).astype(np.float32)).to(self.device)

                # normalize advantages
                if self.params["normalize_advs"]:
                    advs_std = torch.std(advs, unbiased=False)
                    if not advs_std == 0 and not torch.isnan(advs_std):
                        advs = (advs - torch.mean(advs)) / advs_std

                s_start = time.time()

                if not self.params.get('SL_iid', False):
                    # SL loss with BO-like trajectories
                    vpreds, logprobs, entropies, fmodel = self.pi.predict_vals_logps_ents(
                        states_x=states_x, states_y=states_y, states_mask=states_mask, states_ix=states_ix,
                        state_request=state_request, states_mask_shift=states_mask_shift, actions=actions,
                        proba_mask=proba_mask)
                else:
                    # SL loss with i.i.d sampling
                    states_x_sl = torch.zeros_like(states_x)
                    states_y_sl = torch.zeros_like(states_y)
                    states_x_sl[:, state_request[0]:] = states_x[:, state_request[0]:]
                    indexes = torch.randint(low=state_request[0], high=states_x.shape[1],
                                            size=(states_x.shape[0], state_request[0]), device=self.device)

                    ids = torch.arange(states_x.shape[0], device=self.device)
                    states_x_sl[:, :state_request[0]] = states_x[ids[:, None], indexes]
                    states_y_sl[:, :state_request[0]] = y_true[ids[:, None], indexes - state_request[0]]

                    if self.params.get('SL_iid_fast'):
                        states_mask_sl = torch.randint(low=1, high=state_request[0], size=(states_mask.shape[0],),
                                                       device=self.device)

                        # compute values and entropies at current theta, and logprobs at current and old theta
                        _, _, _, fmodel = self.pi.predict_vals_logps_ents(
                            states_x=states_x_sl, states_y=states_y_sl, states_mask=states_mask_sl,
                            states_ix=states_ix, state_request=state_request,
                            states_mask_shift=states_mask_shift, actions=actions, proba_mask=proba_mask)
                        vpreds, logprobs, entropies, _ = self.pi.predict_vals_logps_ents(
                            states_x=states_x, states_y=states_y, states_mask=states_mask, states_ix=states_ix,
                            state_request=state_request, states_mask_shift=states_mask_shift, actions=actions,
                            proba_mask=proba_mask)
                    else:
                        states_x_both = torch.cat((states_x, states_x_sl), 0)
                        states_y_both = torch.cat((states_y, states_y_sl), 0)
                        states_mask_both = torch.cat((states_mask, torch.randint(low=0, high=state_request[0],
                                                                                 size=(states_mask.shape[0],),
                                                                                 device=self.device)), 0)
                        states_ix_both = torch.cat((states_ix, states_ix), 0)
                        actions_both = torch.cat((actions, actions), 0)
                        proba_mask_both = torch.cat((proba_mask, proba_mask), 0)
                        states_mask_shift_both = torch.cat((states_mask_shift, states_mask_shift), 0)

                        # compute values and entropies at current theta, and logprobs at current and old theta
                        vpreds, logprobs, entropies, fmodel = self.pi.predict_vals_logps_ents(
                            states_x=states_x_both, states_y=states_y_both, states_mask=states_mask_both,
                            states_ix=states_ix_both, state_request=state_request,
                            states_mask_shift=states_mask_shift_both, actions=actions_both, proba_mask=proba_mask_both)

                        vpreds = vpreds[:states_x.shape[0]]
                        logprobs = logprobs[:states_x.shape[0]]
                        entropies = entropies[:states_x.shape[0]]
                        fmodel = fmodel[states_x.shape[0]:]

                s_start = time.time()

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

                # cross entropy loss
                if fmodel.shape[-1] == 1:
                    loss_ce = (fmodel - y_true).square().mean()
                    ent_inside = 0
                elif fmodel.shape[-1] == 2:
                    mean = fmodel[:, :, 0:1]
                    sigma = torch.nn.functional.softplus(fmodel[:, :, 1:2])

                    loss_ce = torch.nn.functional.gaussian_nll_loss(mean, y_true, sigma)
                    ent_inside = sigma.detach().mean()
                else:
                    if self.ddp:
                        loss_ce = - self.pi.policy_net.module.buckets(fmodel, y_true.squeeze(-1)).mean()
                    else:
                        loss_ce = - self.pi.policy_net.buckets(fmodel, y_true.squeeze(-1)).mean()

                    probs = fmodel.detach().softmax(-1)
                    ent_inside = (probs * probs.log()).sum(-1).mean()

                loss_reg = 0.
                if self.params.get("covar_reg_dict", None) is not None and self.params["covar_reg_dict"]["coeff"] > 0.:
                    covar_reg_dict = self.params["covar_reg_dict"]
                    x_test = states_x[:, state_request[0]:]  # (b, N, D)
                    xcont_dims = covar_reg_dict.get("xcont_dims", None)
                    xcat_dims = covar_reg_dict.get("xcat_dims", None)
                    if xcont_dims is None and xcat_dims is None:
                        # backward compatibility: if no dimensions are provided, just assume all dims are continuous
                        xcont_dims = list(range(x_test.shape[-1]))
                    xcont_test = x_test[..., xcont_dims]
                    xcat_test = x_test[..., xcat_dims]

                    for i in range(x_test.shape[0]):
                        if xcont_dims is None or len(xcont_dims) == 0:
                            dist_xcont_test_i = 0.
                        else:
                            if covar_reg_dict["x_dist"] == 'inf':
                                dist_xcont_test_i = torch.cdist(xcont_test[i], xcont_test[i], p=torch.inf)
                            elif covar_reg_dict["x_dist"][0] == 'l':
                                p = float(covar_reg_dict["x_dist"][1:])
                                dist_xcont_test_i = torch.cdist(xcont_test[i], xcont_test[i], p=p)
                            else:
                                raise ValueError(covar_reg_dict["x_dist"])
                        if xcat_dims is None or len(xcat_dims) == 0:
                            dist_xcat_test_i = 0.
                        else:
                            dist_xcat_test_i = torch.cdist(xcat_test[i], xcat_test[i], p=0)

                        dist_x_test_i = dist_xcont_test_i + dist_xcat_test_i
                        eps_mask_i = torch.where(dist_x_test_i > covar_reg_dict["eps"], 0, 1)
                        eps_mask_i -= torch.eye(*eps_mask_i.shape[-2:]).to(eps_mask_i)

                        log_post_i = torch.log_softmax(fmodel[i], -1)
                        post_i = torch.softmax(fmodel[i], -1)
                        if covar_reg_dict["hist_dist"] == 'kl':
                            # solution adapted from https://discuss.pytorch.org/t/calculate-p-pair-wise-kl-divergence/131424/2
                            dist_posterior_i = (post_i * log_post_i).sum(dim=1) - torch.einsum('ik,jk->ij', log_post_i, post_i)
                        else:  # kl_reg_dict["hist_dist"] == "l2"
                            dist_posterior_i = torch.cdist(post_i, post_i, p=2)
                        dist_posterior_i_pos_mask = (dist_posterior_i > 0).to(dist_posterior_i)
                        dist_posterior_i_pos = dist_posterior_i_pos_mask * dist_posterior_i

                        weighted_dist_i = dist_posterior_i_pos * (covar_reg_dict["eps"] - dist_x_test_i) / covar_reg_dict["eps"] * eps_mask_i
                        weighted_dist_i_nnz = torch.count_nonzero(weighted_dist_i, dim=-1)
                        reg_i = weighted_dist_i.sum(-1).div(weighted_dist_i_nnz + 1)  # prevent division by zero
                        loss_reg += reg_i.mean()

                loss = 0
                # argmax-safe
                if self.params["ppo_coeff"] != 0.0:
                    loss += self.params["ppo_coeff"] * loss_ppo

                if self.params["ent_coeff"] != 0.0:
                    loss += self.params["ent_coeff"] * loss_ent

                if self.params.get("covar_reg_dict", None) is not None and self.params["covar_reg_dict"]["coeff"] > 0.:
                    loss += self.params["covar_reg_dict"]["coeff"] * loss_reg

                loss += self.params["value_coeff"] * loss_value + \
                        self.params["ce_coeff"] * loss_ce

                with torch.no_grad():
                    loss_ppo_ep += loss_ppo
                    loss_value_ep += loss_value
                    loss_ent_ep += loss_ent
                    loss_ce_ep += loss_ce
                    loss_ep += loss
                    ent_inside_ep += ent_inside

                n_minibatches += 1

                if n_minibatches % self.params["grad_accumulation"] == 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.params["grad_clip"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.stats["n_optsteps"] += 1
                elif self.ddp:
                    with self.pi.policy_net.no_sync():
                        loss.backward()
                else:
                    loss.backward()

                network_process_time += time.time() - s_start
                s_start = time.time()

            loss_ppo_avg = loss_ppo_ep / n_minibatches
            loss_value_avg = loss_value_ep / n_minibatches
            loss_ent_avg = loss_ent_ep / n_minibatches
            loss_ce_ep = loss_ce_ep / n_minibatches
            loss_avg = loss_ep / n_minibatches
            ent_inside_avg = ent_inside_ep / n_minibatches
            self.iter_log_str += "   loss_ppo = {: .4g}, loss_value = {: .4g}, loss_ent = {: .4g}, loss_ce = {: .4g}, loss = {: .4g}\n".format(
                loss_ppo_avg, loss_value_avg, loss_ent_avg, loss_ce_ep, loss_avg)

            self.stats['loss_avg'].append(loss_avg)
            self.stats['loss_ppo_avg'].append(loss_ppo_avg)
            self.stats['loss_ent_avg'].append(loss_ent_avg)
            self.stats['loss_value_avg'].append(loss_value_avg)
            self.stats['loss_ce_avg'].append(loss_ce_ep)
            self.stats['ent_inside_avg'].append(ent_inside_avg)

        t_optim = time.time() - now
        if self.scheduler:
            self.scheduler.step()
        self.iter_log_str += "  lr {:.7f}\n".format(self.scheduler.get_last_lr()[0] if self.scheduler else self.params['lr'])
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
                if self.rank == 0:
                    self.store_weights()

            t_optim = self.optimize_on_batch()
            self.store_log()
            self.stats["t_train"] += t_optim
            self.stats["t_optim"] = t_optim
            self.stats["n_iters"] += 1
        self.batch_recorder.cleanup()

    def test(self, stop_workers=True):
        # precompute seeds and dataset indexes to dispatch to workers
        self.set_all_seeds()
        number_of_task = int(self.params["max_steps"] / gym.spec(self.params['env_id']).max_episode_steps)
        number_of_datasets = len(gym.spec(self.params['env_id']).kwargs['f_opts']['data'])
        assert self.params["max_steps"] % gym.spec(self.params['env_id']).max_episode_steps == 0
        seed_list = np.random.randint(low=0, high=2**32 - 1, size=number_of_task)
        dataset_index_list = [i % number_of_datasets for i in range(number_of_task)]
        self.batch_recorder.dispatch_seeds(seed_list, dataset_index_list)
        assert len(seed_list) == len(dataset_index_list)

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
        if stop_workers:
            self.batch_recorder.cleanup()

    def store_log(self):
        if self.rank == 0:
            with open(os.path.join(self.logpath, "log"), "a") as f:
                print(self.iter_log_str, file=f)
            if not self.verbose:
                print(self.iter_log_str)

        for k in ['n_timesteps', 'n_optsteps', 't_train', 'perc']:
            self.writer.add_scalar(k, self.stats[k], self.stats["n_iters"])
        for k in ['avg_step_reward', 'avg_initial_reward', 'avg_terminal_reward', 'avg_ep_reward', 'avg_ep_len', 't_batch', 'sps', 'regret_avg']:
            self.writer.add_scalar(k, self.stats['batch_stats'][k], self.stats["n_iters"])
        if 'loss_avg' in self.stats:
            for k in ['loss_avg', 'loss_ppo_avg', 'loss_ent_avg', 'loss_value_avg', 'loss_ce_avg', 'ent_inside_avg']:
                for val in self.stats[k]:
                    self.writer.add_scalar(k, val, self.stats["n_iters"])
        self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0] if self.scheduler else self.params['lr'], self.stats["n_iters"])
        self.writer.flush()

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

    def store_weights(self):
        with open(os.path.join(self.logpath, "weights_" + str(self.stats["n_iters"])), "wb") as f:
            torch.save(self.pi.state_dict(), f)
        with open(os.path.join(self.logpath, "optim_weights_" + str(self.stats["n_iters"])), "wb") as f:
            torch.save(self.optimizer.state_dict(), f)

    def add_iter_log_str(self, batch_stats):
        self.iter_log_str += "\n******************** ITERATION {:2d} ********************\n".format(self.stats["n_iters"])
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
