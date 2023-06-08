# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import multiprocessing as mp

import sys
from pathlib import Path
root = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path.insert(0, root)

from nap.environment.antigen import get_antigen_datasets
from nap.policies.transformer import generate_D_q_matrix

if __name__ == '__main__':
    mp.set_start_method('spawn')

    import torch
    from datetime import datetime
    from nap.RL.ppo_nap import PPO_NAP
    from nap.policies.nap import NAP
    from gym.envs.registration import register
    import torch.distributed as dist

    ddp = False
    if len(sys.argv) > 1:
        ddp = bool(sys.argv[1])

    print("Enable DDP? ", ddp)
    if ddp:
        dist.init_process_group("nccl")

    datasets, trained_gps, _, _, _, _ = get_antigen_datasets(root)

    # specifiy environment
    env_spec = {
        "env_id": "MixNAP-SLiidRL-MLP-CondP-v0",
        "f_type": "condAntigen",
        "D": 11,
        "f_opts": {
            "min_regret": 1e-20,
            "models": trained_gps,
            "data": datasets,
            "cat_dims": list(range(11)),
            "cat_alphabet": {c: list(range(22)) for c in list(range(11))},
            "cat_alphabet_map": None,
            "cont_dims": [],
            "perturb_training_inputs": True,
            "nb_perturbed_pos": 1,
            "shuffle_and_cutoff": False,
        },
        "features": ["incumbent", "timestep_perc"],
        "T": 24,
        "n_init_samples": 0,
        "pass_X_to_pi": False,
        "local_af_opt": False,
        "cardinality_domain": 1000,
        "reward_transformation": "neg_log10",  # neg_log10, best_regret_timed
        "use_index_speedup": True,
    }

    low_memory = True
    print("Low memory profile:", low_memory)

    # specify PPO parameters
    n_iterations = 2000
    batch_size = 1440 // (dist.get_world_size() if ddp else 1)
    n_workers = 5  # collecting workers per GPUs
    if low_memory:
        n_workers = 3 if ddp else 5

    y_range = (-0.1, 1.1)
    print('y_range:', y_range)
    arch_spec = dict(nbuckets=1000, dim_feedforward=1024, emb_size=512, nlayers=6, nhead=4, dropout=0.0,
                     temperature=0.1, y_range=y_range,
                     af_name="mlp" if "MLP" in env_spec["env_id"] else ("ucb" if "UCB" in env_spec["env_id"] else "ei"),
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
        "ppo_coeff": 1.0 if "RL" in env_spec["env_id"] else 0.0,
        "value_coeff": 1.0,
        "ent_coeff": 0.0,
        "ce_coeff": 1.0,

        "SL_iid": False,
        "SL_iid_fast": True,

        "decay_lr": True,
        "gamma": 0.98,
        "lambda": 0.98,
        "grad_clip": 0.5,
        "loss_type": "GAElam",
        "normalize_advs": True,
        "n_workers": n_workers,
        "env_id": env_spec["env_id"],
        "seed": 0,
        "argmax": False,
        "env_seeds": list(range(n_workers)),
        "policy_options": {
            "max_query": env_spec["cardinality_domain"],
            "arch_spec": arch_spec,
            "use_value_network": True,
            "arch_spec_value": arch_spec
        },
        "covar_reg_dict": {
            "coeff": 1.0 if "KL" in env_spec["env_id"] else 0.0,
            "xcont_dims": env_spec["f_opts"]["cont_dims"],
            "xcat_dims": env_spec["f_opts"]["cat_dims"],
            "x_dist": "inf",
            "hist_dist": "kl",
            "eps": 2.0,  # for cat dimensions, the distance is hamming (L0)
        },
    }

    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="nap.environment.function_gym_nap:NAPEnv",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # log data and weights go here, use this folder for evaluation afterwards
    logpath = os.path.join(root, "log", 'TRAIN/Antigen', env_spec["env_id"],
                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

    if env_spec["f_opts"].get("shuffle_and_cutoff", False) or env_spec["f_type"] == "condAntigen":
        # pre-compute masks for computational gain
        policy_net_masks = []
        for i in range(env_spec["T"]):
            policy_net_masks.append(
                generate_D_q_matrix(env_spec["T"] + env_spec["cardinality_domain"] + env_spec["n_init_samples"],
                                    env_spec["n_init_samples"] + i))
        policy_net_masks = torch.stack(policy_net_masks)
    else:
        # we have datasets of different sizes, so we can't pre-compute masks using env_spec["cardinality_domain"].
        # We set them to None and will let the forward pass figure out their shapes
        policy_net_masks = None

    # set up policy
    policy_fn = lambda observation_space, action_space, deterministic, dataparallel: NAP(
        observation_space=observation_space,
        action_space=action_space,
        deterministic=True if ppo_spec["argmax"] else deterministic,
        options=ppo_spec["policy_options"],
        dataparallel=dataparallel,
        policy_net_masks=policy_net_masks,
        mixed_type_options={
            "cat_dims": env_spec["f_opts"]["cat_dims"],
            "num_dims": env_spec["f_opts"]["cont_dims"],
            "cat_alphabet": env_spec["f_opts"]["cat_alphabet"],
            "cat_alphabet_map": env_spec["f_opts"]["cat_alphabet_map"],
            "mixed_type": "MixNAP" in env_spec["env_id"],
        },
    )

    # do training
    if not ddp or dist.get_rank() == 0:
        print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))

    ppo = PPO_NAP(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=100)
    ppo.train()
