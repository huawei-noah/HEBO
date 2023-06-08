# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import multiprocessing as mp
import pickle

import numpy as np

from nap.environment.hpo import get_hpo_specs, get_cond_hpo_specs
from nap.policies.fsbo import FSBO

if __name__ == '__main__':
    mp.set_start_method('spawn')

    from datetime import datetime
    from nap.RL.ppo_nap import PPO_NAP
    from nap.policies.policies import FSBO_EI
    from gym.envs.registration import register

    ddp = False
    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")

    hpo_type = "hpobenchXGB"  # pm25, oil, augment, hpobenchXGB, Asteroid
    dims, points, train_datasets, valid_datasets, test_datasets, kernel_lengthscale, kernel_variance, \
    noise_variance, X_mean, X_std = get_hpo_specs(hpo_type)

    _, _, train_gp_models, _, _, _, _, _ = get_cond_hpo_specs(hpo_type)

    # specifiy environment
    env_spec = {
        "env_id": f"FSBO-T295-v0",
        "f_type": "HPO",
        "D": dims,
        "f_opts": {
            "min_regret": 1e-20,
            "data": test_datasets,
            "X_mean": X_mean,
            "X_std": X_std,
        },
        "features": ["incumbent", "timestep_perc"],
        "T": 295,
        "n_init_samples": 5,
        "pass_X_to_pi": False,
        "local_af_opt": False,
        "cardinality_domain": points,
        "reward_transformation": "neg_log10",  # true maximum not known
        "use_index_speedup": True,
    }

    # log data and weights go here, use this folder for evaluation afterwards
    logpath = os.path.join(rootdir, "log/TEST", hpo_type, env_spec["env_id"],
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

    # format data for FSBO
    train_data = {str(i): dict(X=data['domain'].tolist(), y=data["accs"][:, None].tolist())
                  for i, path in enumerate(train_datasets)
                  for data in [pickle.load(open(path, "rb"))]}

    valid_data = {str(i): dict(X=data['domain'].tolist(), y=data["accs"][:, None].tolist())
                  for i, path in enumerate(valid_datasets)
                  for data in [pickle.load(open(path, "rb"))]}

    # FSBO training
    fsbo_model = FSBO(train_data=train_data, valid_data=valid_data, checkpoint_path=logpath)
    print("Training FSBO for 10k epochs")
    fsbo_model.meta_train(epochs=10000)
    fsbo_model.to('cpu')


    # specify PPO parameters
    n_iterations = 1
    n_seeds = 10
    batch_size = env_spec['T'] * n_seeds * len(test_datasets)
    n_workers = 5  # collecting workers per GPUs

    ppo_spec = {
        "batch_size": batch_size,
        "max_steps": n_iterations * batch_size,
        "n_workers": n_workers,
        "env_id": env_spec["env_id"],
        "seed": 0,
        "argmax": True,
        "env_seeds": list(range(n_workers)),
        "gamma": 0.98,
        "lambda": 0.98,
        "lr": 3e-5,
        "finetune": True,  # for testing FSBO, need to fine tune the ExactGPLayer (along with the feature extractor)
        "policy_options": {
            "max_query": env_spec["cardinality_domain"],
            "use_value_network": False,
        }
    }

    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym_nap:NAPEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # set up policy
    policy_fn = lambda observation_space, action_space, deterministic, dataparallel: FSBO_EI(logpath, dims, ppo_spec['finetune'])

    # do testing
    print("Testing on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))

    ppo = PPO_NAP(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    ppo.test()

    test_reward = np.array(ppo.teststats['avg_ep_reward']).mean(), np.array(ppo.teststats['avg_ep_reward']).std()
    test_regret = np.array(ppo.teststats['regret']).mean(), np.array(ppo.teststats['regret']).std()

    print('======================================================')
    print('======================= DONE =========================')
    print('======================================================')
    print("========== REWARD ==========")
    print(f'mean={test_reward[0]:.5f} std={test_reward[1]:.5f}')
    print("========== REGRET ==========")
    print(f'mean={test_regret[0]:.5f} std={test_regret[1]:.5f}')

    statsfile = os.path.join(logpath, 'test_results.txt')
    with open(statsfile, 'w') as f:
        f.write("========== REWARD ==========\n")
        f.write(f'mean={test_reward[0]:.5f} std={test_reward[1]:.5f}\n')
        f.write("========== REGRET ==========\n")
        f.write(f'mean={test_regret[0]:.5f} std={test_regret[1]:.5f}\n')
