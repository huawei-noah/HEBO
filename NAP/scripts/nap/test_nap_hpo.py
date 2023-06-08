# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import sys
import multiprocessing as mp

import numpy as np

from nap.environment.hpo import get_hpo_specs
from nap.policies.transformer import generate_D_q_matrix

if __name__ == '__main__':
    mp.set_start_method('spawn')

    import torch
    from datetime import datetime
    from nap.RL.ppo_nap import PPO_NAP
    from nap.policies.nap import NAP
    from gym.envs.registration import register
    import torch.distributed as dist
    import socket

    ddp = False
    if len(sys.argv) > 1:
        ddp = bool(sys.argv[1])

    print("Enable DDP? ", ddp)
    if ddp:
        dist.init_process_group("nccl")

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    rootdir_nap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nap")

    hpo_type = "hpobenchXGB"  # pm25, oil, augment, hpobenchXGB, Asteroid
    num_dims, cat_dims, _, _, _, points, train_datasets, valid_datasets, test_datasets, _, _, _, X_mean, X_std = get_hpo_specs(hpo_type)

    saved_model_path = ""

    # specifiy environment
    env_spec = {
        "env_id": f"NAP-SLiid-v0",
        "f_type": "HPO",
        "D": len(num_dims) + len(cat_dims),
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

    low_memory = True
    print("Low memory profile:", low_memory)

    # specify PPO parameters
    n_iterations = 5
    batch_size = 1440 // (dist.get_world_size() if ddp else 1)
    n_workers = 1

    y_range = (-0.1, 1.1)
    print('y_range:', y_range)
    arch_spec = dict(nbuckets=1000, dim_feedforward=1024, emb_size=512, nlayers=6, nhead=4, dropout=0.0,
                     temperature=0.1, y_range=y_range,
                     af_name="mlp" if "MLP" in env_spec["env_id"] else (
                         "ucb" if "UCB" in env_spec["env_id"] else "ei"),
                     joint_model_af_training=False if "Disjoint" in env_spec["env_id"] else True,
                     )

    ppo_spec = {
        "batch_size": batch_size,
        "max_steps": n_iterations * batch_size,
        "minibatch_size": 16 if not low_memory else 8,
        "grad_accumulation": 2 if not low_memory else 4,
        "n_epochs": 1,
        "lr": 3e-5,
        "epsilon": 0.15,
        "ppo_coeff": 0.0,
        "value_coeff": 1.0,
        "ent_coeff": 0.0,
        "ce_coeff": 1.0,
        "decay_lr": True,
        "gamma": 0.98,
        "lambda": 0.98,
        "grad_clip": 0.5,
        "loss_type": "GAElam",
        "normalize_advs": True,
        "n_workers": n_workers,
        "env_id": env_spec["env_id"],
        "seed": 0,
        "argmax": True,
        "env_seeds": list(range(n_workers)),
        "policy_options": {
            "max_query": env_spec["cardinality_domain"],
            "arch_spec": arch_spec,
            "use_value_network": True,
            "arch_spec_value": arch_spec
        }
    }

    # after Model pretraining
    ppo_spec.update({
        "load": True,
        "load_path": saved_model_path,
        "param_iter": "CKPT_TO_FILL",
    })

    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym_nap:NAPEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # log data and weights go here, use this folder for evaluation afterwards
    logpath = os.path.join(rootdir_nap, "log/TEST/", hpo_type, env_spec["env_id"],
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

    # build masks once and for all
    policy_net_masks = []
    for i in range(env_spec["T"]):
        policy_net_masks.append(generate_D_q_matrix(env_spec["T"] + env_spec["cardinality_domain"] +
                                                    env_spec["n_init_samples"],
                                                    env_spec["n_init_samples"] + i))
    policy_net_masks = torch.stack(policy_net_masks)
    if hpo_type in ["pm25", "hpobenchXGB", "oil"]:
        policy_net_masks = None

    # set up policy
    policy_fn = lambda observation_space, action_space, deterministic, dataparallel: NAP(observation_space=observation_space,
                                                                                         action_space=action_space,
                                                                                         deterministic=True if ppo_spec["argmax"] else deterministic,
                                                                                         options=ppo_spec["policy_options"],
                                                                                         dataparallel=dataparallel,
                                                                                         policy_net_masks=policy_net_masks)

    # do testing
    if not ddp or dist.get_rank() == 0:
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
