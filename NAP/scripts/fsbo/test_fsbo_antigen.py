# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from nap.environment.antigen import get_antigen_datasets
from nap.policies.fsbo import FSBO

if __name__ == '__main__':
    mp.set_start_method('spawn')

    from datetime import datetime
    from nap.RL.ppo_nap import PPO_NAP
    from nap.policies.policies import FSBO_EI
    from gym.envs.registration import register

    ddp = False
    rootdir_nap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")
    datasets_train, gps_train, datasets_val, gps_val, datasets_test, gps_test = get_antigen_datasets("./")

    gp_params = {
        "lengthscale": [],
        "mean_const": [],
        "lik_noise": [],
    }
    for gp_path in gps_train:
        import sys
        import nap.RL.utils_gp
        sys.modules['fsaf'] = sys.modules['nap']
        sys.modules['fsaf.RL.gp_utils'] = sys.modules['nap.RL.utils_gp']
        gp_model = torch.load(gp_path, map_location='cpu')
        gp_params["lengthscale"].append(gp_model.model.covar_module.lengthscale.detach().cpu().numpy())
        gp_params["lik_noise"].append(gp_model.likelihood.noise.detach().cpu().numpy())
        gp_params["mean_const"].append(gp_model.model.mean_module.constant.detach().cpu().numpy())

    lengthscale = np.concatenate(gp_params["lengthscale"]).mean(0).tolist()
    lik_noise = np.concatenate(gp_params["lik_noise"]).mean().item()
    mean_const = np.array(gp_params["mean_const"]).mean().item()

    # specifiy environment
    env_spec = {
        "env_id": f"FSBO-T295-v0",
        "f_type": "Antigen",
        "D": 11,
        "f_opts": {
            "kernel": "Categorical",
            "min_regret": 1e-20,
            "data": datasets_test,
            "cat_dims": [0,1,2,3,4,5,6,7,8,9,10],
            "num_classes": [20,20,20,20,20,20,20,20,20,20,20],
            "cont_dims": None,
            "lengthscale": lengthscale,
            "likelihood_noise": lik_noise,
            "mean_constant": mean_const,
            "shuffle_and_cutoff": False,
        },
        "features": ["incumbent", "timestep_perc"],
        "T": 295,
        "n_init_samples": 5,
        "pass_X_to_pi": False,
        "local_af_opt": False,
        "cardinality_domain": 1000,
        "reward_transformation": "neg_log10",  # true maximum not known
        "use_index_speedup": True,
    }

    # log data and weights go here, use this folder for evaluation afterwards
    logpath = os.path.join(rootdir_nap, "log/TEST", 'Antigen', env_spec["env_id"],
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

    # format data for FSBO
    def clean(seq_string):
        return list(map(int, seq_string.split(',')))


    train_data = {}
    for i, path in enumerate(datasets_train):
        data = pd.read_csv(path, converters={'domain': clean})
        train_data[str(i)] = dict(X=data['domain'].values.tolist(), y=data["accs"].values[:, None].tolist())

    valid_data = {}
    for i, path in enumerate(datasets_val):
        data = pd.read_csv(path, converters={'domain': clean})
        valid_data[str(i)] = dict(X=data['domain'].values.tolist(), y=data["accs"].values[:, None].tolist())

    # FSBO training
    fsbo_model = FSBO(train_data=train_data, valid_data=valid_data, checkpoint_path=logpath,
                      cat_idx=env_spec["f_opts"]["cat_dims"], num_classes=env_spec["f_opts"]["num_classes"])
    print("Training FSBO for 10k epochs")
    fsbo_model.meta_train(epochs=10000)
    fsbo_model.to('cpu')


    # specify PPO parameters
    n_iterations = 1
    n_seeds = 10
    batch_size = env_spec['T'] * n_seeds * len(datasets_test)
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
    policy_fn = lambda observation_space, action_space, deterministic, dataparallel:\
        FSBO_EI(logpath, env_spec["D"], ppo_spec['finetune'],
                cat_idx=env_spec["f_opts"]["cat_dims"],
                num_classes=env_spec["f_opts"]["num_classes"])

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
