# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
from datetime import datetime

import gym
import numpy as np

from nap.environment.antigen import get_antigen_datasets
from nap.policies.policies import UCB, EI, RS
from nap.RL.ppo import PPO
from gym.envs.registration import register
import torch

if __name__ == '__main__':
    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    rootdir_nap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")
    _, gps_train, _, _, test_datasets, _ = get_antigen_datasets(rootdir)

    gp_params = {
        "lengthscale": [],
        "mean_const": [],
        "lik_noise": [],
    }
    for gp_path in gps_train:
        gp_model = torch.load(gp_path, map_location='cpu')
        gp_params["lengthscale"].append(gp_model.model.covar_module.lengthscale.detach().cpu().numpy())
        gp_params["lik_noise"].append(gp_model.likelihood.noise.detach().cpu().numpy())
        gp_params["mean_const"].append(gp_model.model.mean_module.constant.detach().cpu().numpy())

    lengthscale = np.concatenate(gp_params["lengthscale"]).mean(0).tolist()
    lik_noise = np.concatenate(gp_params["lik_noise"]).mean().item()
    mean_const = np.array(gp_params["mean_const"]).mean().item()

    # TODO online std

    # specifiy environment
    env_spec = {
        "env_id": f"GP-CatKern-T295-v0",
        "f_type": "Antigen",
        "D": 11,
        "f_opts": {
            "kernel": "Categorical",
            "min_regret": 1e-20,
            "data": test_datasets,
            "cat_dims": [0,1,2,3,4,5,6,7,8,9,10],
            "num_classes": [20,20,20,20,20,20,20,20,20,20,20],
            "cont_dims": None,
            "lengthscale": lengthscale,
            "likelihood_noise": lik_noise,
            "mean_constant": mean_const,
        },
        "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"],
        "T": 295,
        "n_init_samples": 5,
        "pass_X_to_pi": False,
        "local_af_opt": False,
        "cardinality_domain": 1000,
        "remove_seen_points": True,  # only True for testing
        # will be set individually for each new function to the sampled hyperparameters
        "kernel": "Categorical",
        "kernel_lengthscale": None,  # kernel_lengthscale,
        "kernel_variance": None,  # kernel_variance,
        "noise_variance": None,  # noise_variance,
        "use_prior_mean_function": False,
        "reward_transformation": "neg_log10"  # true maximum not known
    }

    # specify PPO parameters
    # 1 iteration will run n_seeds seeds so e.g. running 5 iterations with n_seeds=10 will run 50 seeds per test dataset
    n_iterations = 1
    n_workers = 40
    n_seeds = 10  # number of seeds per test task and per iteration
    batch_size = len(test_datasets) * env_spec["T"] * n_seeds
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

    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym:MetaBOEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    dummy_env = gym.make(env_spec["env_id"])
    feature_order = dummy_env.unwrapped.feature_order_eval_envs
    D = dummy_env.unwrapped.D
    # # set up policy
    # policy_fn = lambda obs, acspace, deterministic: UCB(feature_order=feature_order, kappa="gp_ucb", D=D, delta=0.0001)
    #
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-fixed-UCB",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")

    # policy_fn = lambda obs, acspace, deterministic: EI(feature_order=feature_order)
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-fixed-EI",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")
    #
    # policy_fn = lambda obs, acspace, deterministic: RS()
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-fixed-RS",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")

    print("WITHOUT PRIOR")
    env_spec['update_gp'] = True
    del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    # # register environment
    # register(
    #     id=env_spec["env_id"],
    #     entry_point="nap.environment.function_gym:MetaBOEnv",
    #     max_episode_steps=env_spec["T"],
    #     reward_threshold=None,
    #     kwargs=env_spec
    # )
    #
    # # set up policy
    # policy_fn = lambda obs, acspace, deterministic: UCB(feature_order=feature_order, kappa="gp_ucb", D=D, delta=0.0001)
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-learnt-UCB",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")
    #
    # del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym:MetaBOEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )
    policy_fn = lambda obs, acspace, deterministic: EI(feature_order=feature_order)
    logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-learnt-EI",
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # do testing
    print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    ppo.test()
    test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    print(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    print(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    statsfile = os.path.join(logpath, 'test_results.txt')
    with open(statsfile, 'w') as f:
        f.write(f"=== REWARD ===\nmean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
        f.write(f"=== REGRET ===\nmean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")

    print("RANDOM INIT AND UPDATED")
    env_spec['update_gp'] = True
    env_spec['random_init_gp'] = True
    del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    # # register environment
    # register(
    #     id=env_spec["env_id"],
    #     entry_point="nap.environment.function_gym:MetaBOEnv",
    #     max_episode_steps=env_spec["T"],
    #     reward_threshold=None,
    #     kwargs=env_spec
    # )
    #
    # # set up policy
    # policy_fn = lambda obs, acspace, deterministic: UCB(feature_order=feature_order, kappa="gp_ucb", D=D, delta=0.0001)
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-learnt-rinit-UCB",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")
    #
    # del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    # # register environment
    # register(
    #     id=env_spec["env_id"],
    #     entry_point="nap.environment.function_gym:MetaBOEnv",
    #     max_episode_steps=env_spec["T"],
    #     reward_threshold=None,
    #     kwargs=env_spec
    # )
    # policy_fn = lambda obs, acspace, deterministic: EI(feature_order=feature_order)
    # logpath = os.path.join(rootdir_nap, "log/TEST", "Antigen", env_spec["env_id"] + "-learnt-rinit-EI",
    #                        datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # # do testing
    # print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    # ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    # ppo.test()
    # test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    # test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    # print(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    # print(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    # statsfile = os.path.join(logpath, 'test_results.txt')
    # with open(statsfile, 'w') as f:
    #     f.write(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
    #     f.write(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")
