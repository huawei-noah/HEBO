# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import multiprocessing as mp

import gym
import numpy as np

from nap.environment.hpo import get_hpo_specs, get_cond_hpo_specs

if __name__ == '__main__':
    mp.set_start_method('spawn')

    from datetime import datetime
    from nap.RL.ppo import PPO
    from nap.policies.policies import iclr2020_NeuralAF
    from gym.envs.registration import register

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")

    hpo_type = "svm"  # pm25, oil, augment, hpobenchXGB, Asteroid
    dims, points, train_datasets, valid_datasets, test_datasets, kernel_lengthscale, kernel_variance, \
    noise_variance, X_mean, X_std = get_hpo_specs(hpo_type)

    kernel = "RBF"

    reward = {}
    regret = {}
    for ckpt in range(100, 2100, 100):
        ckpt -= 1 if ckpt == 2000 else 0

        # specifiy environment
        env_spec = {
            "env_id": f"MetaBO-T295-fixed-v0",
            "f_type": "HPO",
            "D": dims,
            "f_opts": {
                "min_regret": 1e-20,
                "data": valid_datasets,
                "X_mean": X_mean,
                "X_std": X_std,
            },
            "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"],
            "remove_seen_points": True,  # only True for testing
            "T": 295,
            "n_init_samples": 5,
            "pass_X_to_pi": False,

            # will be set individually for each new function to the sampled hyperparameters
            "kernel": kernel,
            "kernel_lengthscale": kernel_lengthscale,
            "kernel_variance": kernel_variance,
            "noise_variance": noise_variance,
            "use_prior_mean_function": False,

            "local_af_opt": False,
            "cardinality_domain": points,
            "reward_transformation": "neg_log10",  # true maximum not known
        }

        # specify PPO parameters
        # 1 iteration will run n_seeds seeds so e.g. running 5 iterations with n_seeds=10 will run 50 seeds per test dataset
        n_iterations = 5
        n_workers = 5
        n_seeds = 10  # number of seeds per test task
        batch_size = len(valid_datasets) * env_spec["T"] * n_seeds

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
            "load_path": f"nap/log/TRAIN/svm/MetaBO-fixed-v0/2023-03-28-19-25-22/",
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
        logpath = os.path.join(rootdir, "log/VALIDATE/", hpo_type, env_spec["env_id"], f"{ckpt}_ckpt", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

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
    statsfile = os.path.join(rootdir, "log/VALIDATE/", hpo_type, env_spec["env_id"], 'validation.txt')
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