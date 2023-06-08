# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import pickle
import sys
import multiprocessing as mp
import gym
import numpy as np

from nap.environment.antigen import get_antigen_datasets
import torch

if __name__ == '__main__':
    mp.set_start_method('spawn')

    from datetime import datetime
    from nap.RL.ppo import PPO
    from nap.policies.policies import iclr2020_NeuralAF
    from gym.envs.registration import register

    torch.cuda.set_device(0)

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    rootdir_nap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")
    _, _, datasets_val, trained_gps_val, _, _ = get_antigen_datasets(rootdir)

    gp_params = {
        "lengthscale": [],
        "mean_const": [],
        "lik_noise": [],
    }
    for gp_path in trained_gps_val:
        gp_model = torch.load(gp_path, map_location='cpu')
        gp_params["lengthscale"].append(gp_model.model.covar_module.lengthscale.detach().cpu().numpy())
        gp_params["lik_noise"].append(gp_model.likelihood.noise.detach().cpu().numpy())
        gp_params["mean_const"].append(gp_model.model.mean_module.constant.detach().cpu().numpy())

    lengthscale = np.concatenate(gp_params["lengthscale"]).mean(0).tolist()
    lik_noise = np.concatenate(gp_params["lik_noise"]).mean().item()
    mean_const = np.array(gp_params["mean_const"]).mean().item()

    reward = {}
    regret = {}
    for ckpt in range(100, 2100, 100):
        ckpt -= 1 if ckpt == 2000 else 0

        # specifiy environment
        env_spec = {
            "env_id": f"MetaBO-CatKern-fixed-L288-v0",
            "f_type": "Antigen",
            "D": 11,
            "f_opts": {
                "kernel": "Categorical",
                "min_regret": 1e-20,
                "data": datasets_val,
                "cat_dims": [0,1,2,3,4,5,6,7,8,9,10],
                "num_classes": [20,20,20,20,20,20,20,20,20,20,20],
                "cont_dims": None,
                "perturb_training_inputs": False,
                "shuffle_and_cutoff": False,
                "lengthscale": lengthscale,
                "likelihood_noise": lik_noise,
                "mean_constant": mean_const,
            },
            "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"],
            "remove_seen_points": True,  # only True for testing
            "T": 288,
            "n_init_samples": 5,
            "pass_X_to_pi": False,

            # will be set individually for each new function to the sampled hyperparameters
            "kernel": "Categorical",
            "kernel_lengthscale": None,
            "kernel_variance": None,
            "noise_variance": None,
            "use_prior_mean_function": False,

            "local_af_opt": False,
            "cardinality_domain": 300,
            "reward_transformation": "neg_log10",  # true maximum not known
        }

        # specify PPO parameters
        n_iterations = 5
        batch_size = 1440 
        n_workers = 5  # collecting workers per GPUs

        arch_spec = 4 * [200]
        ppo_spec = {
            "batch_size": batch_size,
            "max_steps": n_iterations * batch_size,
            "minibatch_size": batch_size // 20,
            "n_epochs": 4,
            "lr": 1e-4,
            "epsilon": 0.15,
            "value_coeff": 1.0,
            "ent_coeff": 0.0,
            "gamma": 0.98,
            "lambda": 0.98,
            "loss_type": "GAElam",
            "normalize_advs": True,
            "n_workers": n_workers,
            "env_id": env_spec["env_id"],
            "seed": 0,
            "argmax": True,
            "env_seeds": list(range(n_workers)),
            "policy_options": {
                "activations": "relu",
                "arch_spec": arch_spec,
                "exclude_t_from_policy": True,
                "exclude_T_from_policy": True,
                "use_value_network": True,
                "t_idx": -2,
                "T_idx": -1,
                "arch_spec_value": arch_spec
            }
        }

        # after Model pretraining
        ppo_spec.update({
            "load": True,
            "load_path": f"nap/log/TRAIN/Antigen/MetaBO-CatKern-fixed-v0/2023-03-08-17-01-02/",
            "param_iter": str(ckpt),
        })

        # register environment
        register(
            id=env_spec["env_id"],
            entry_point="nap.environment.function_gym:MetaBOEnv",
            max_episode_steps=env_spec["T"],
            reward_threshold=None,
            kwargs=env_spec
        )

        # log data and weights go here, use this folder for evaluation afterwards
        logpath = os.path.join(rootdir_nap, "log/VALIDATE/Antigen", env_spec["env_id"], f"{ckpt}_ckpt",
                               datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
        # set up policy
        policy_fn = lambda observation_space, action_space, deterministic: iclr2020_NeuralAF(observation_space=observation_space,
                                                                                    action_space=action_space,
                                                                                    deterministic=True if ppo_spec["argmax"] else deterministic,
                                                                                    options=ppo_spec["policy_options"])

        # do testing
        print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))

        ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
        ppo.test()

        reward[ckpt] = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
        regret[ckpt] = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
        del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    print('======================================================')
    print('======================= DONE =========================')
    print('======================================================')
    print("========== REWARD ==========")
    for k in reward:
        print(f'ckpt={k} mean={reward[k][0]:.5f} std={reward[k][1]:.5f}')
    print("========== REGRET ==========")
    for k in regret:
        print(f'ckpt={k} mean={regret[k][0]:.5f} std={regret[k][1]:.5f}')

    mean_regrets = np.array([regret[k][0] for k in regret])
    mean_regrets_ckpt = np.array([k for k in regret])
    print(f"Best regret ckpt. is {mean_regrets_ckpt[np.argmin(mean_regrets)]}")
    print(f"Best Regret mean={regret[mean_regrets_ckpt[np.argmin(mean_regrets)]][0]:.5f} "
          f"std={regret[mean_regrets_ckpt[np.argmin(mean_regrets)]][1]:.5f}")
    statsfile = os.path.join(rootdir_nap, "log/VALIDATE/", "Antigen", env_spec["env_id"], 'validation.txt')
    with open(statsfile, 'w') as f:
        f.write("========== REWARD ==========\n")
        for k in reward:
            f.write(f'ckpt={k} mean={reward[k][0]:.5f} std={reward[k][1]:.5f}\n')
        f.write("========== REGRET ==========\n")
        for k in regret:
            f.write(f'ckpt={k} mean={regret[k][0]:.5f} std={regret[k][1]:.5f}\n')
        f.write(f"\nBest regret ckpt. is {mean_regrets_ckpt[np.argmin(mean_regrets)]}\n")
        f.write(f"Best Regret mean={regret[mean_regrets_ckpt[np.argmin(mean_regrets)]][0]:.5f} "
                f"std={regret[mean_regrets_ckpt[np.argmin(mean_regrets)]][1]:.5f}")
