# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import pickle
from datetime import datetime

import gym
import numpy as np
import multiprocessing as mp

from nap.RL.util import print_best_gp_params
from nap.environment.hpo import get_hpo_specs, get_cond_hpo_specs
from nap.environment.objectives import get_HPO_domain
from nap.policies.policies import UCB, EI, RS
from nap.RL.ppo import PPO
from gym.envs.registration import register
import torch

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # torch.cuda.set_device(0)

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")

    hpo_type = "hpobenchXGB"  # pm25, oil, augment, hpobenchXGB, Asteroid
    dims, points, train_datasets, valid_datasets, test_datasets, kernel_lengthscale, kernel_variance, \
    noise_variance, X_mean, X_std = get_hpo_specs(hpo_type)

    _, _, train_gp_models, _, _, _, _, _ = get_cond_hpo_specs(
        hpo_type, root_dir=os.path.dirname(os.path.realpath(__file__)))

    # get GP params
    # loaded_datasets = [pickle.load(open(dataset, "rb")) for dataset in train_datasets + valid_datasets]
    # all_X = np.array([get_HPO_domain(data=dataset) for dataset in loaded_datasets])
    # all_Y = np.array([dataset['accs'] for dataset in loaded_datasets])
    # print_best_gp_params(all_X, all_Y)

    # specifiy environment
    kernel = "Matern"
    env_spec = {
        "env_id": f"GP-T295-v0",
        "f_type": "HPO",
        "D": dims,
        "f_opts": {
            "min_regret": 1e-20,
            "data": test_datasets,
            "X_mean": X_mean,
            "X_std": X_std,
        },
        "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"],
        "T": 295,
        # "T": 100,
        "remove_seen_points": True,  # only True for testing
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
        "reward_transformation": "neg_log10"  # true maximum not known
    }

    # specify PPO parameters
    # 1 iteration will run n_seeds seeds so e.g. running 5 iterations with n_seeds=10 will run 50 seeds per test dataset
    n_iterations = 1
    n_workers = 2
    n_seeds = 10  # number of seeds per test task
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
    # set up policy
    # policy_fn = lambda obs, acspace, deterministic: UCB(feature_order=feature_order, kappa="gp_ucb", D=D, delta=0.0001)
    #
    # logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-fixed-UCB",
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

    policy_fn = lambda obs, acspace, deterministic: EI(feature_order=feature_order)
    logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-fixed-EI",
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

    policy_fn = lambda obs, acspace, deterministic: RS()
    logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-fixed-RS",
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

    print("WITHOUT PRIOR")
    env_spec['update_gp'] = True
    del gym.envs.registration.registry.env_specs[env_spec["env_id"]]

    # register environment
    # register(
    #     id=env_spec["env_id"],
    #     entry_point="nap.environment.function_gym:MetaBOEnv",
    #     max_episode_steps=env_spec["T"],
    #     reward_threshold=None,
    #     kwargs=env_spec
    # )

    # set up policy
    # policy_fn = lambda obs, acspace, deterministic: UCB(feature_order=feature_order, kappa="gp_ucb", D=D, delta=0.0001)
    # logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-learnt-UCB",
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
    logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-learnt-EI",
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
    # logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-learnt-rinit-UCB",
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
    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym:MetaBOEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )
    policy_fn = lambda obs, acspace, deterministic: EI(feature_order=feature_order)
    logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"] + "-learnt-rinit-EI",
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    # do testing
    print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    ppo.test()
    test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()
    print(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}")
    print(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}")
    statsfile = os.path.join(logpath, 'test_results.txt')
    with open(statsfile, 'w') as f:
        f.write(f"=== REWARD === mean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n")
        f.write(f"=== REGRET === mean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n")
