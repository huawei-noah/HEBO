# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import os
import numpy as np
import yaml
from ray import tune
from ray.tune import run

from agents.common.utils import get_global_name, get_global_agent_name
from sac_main_fn import main as sac_main

os.environ["Timer"] = '1'


def trial_name(trial, hp_to_write):
    ti = 'repeat_run'
    identifier = ','.join([f'{hp}={trial.config[hp]}' for hp in hp_to_write]) + \
                 f',trial={trial.config[ti]},id={trial.trial_id}'
    return identifier


if __name__ == '__main__':

    envs = ['ball_in_cup']
    agents = [
        'SAC',
        # 'SAG',
        # 'PIG',
        # 'PAG',
    ]
    nb_local_experts = 'simple'

    for env in envs:

        glob_name = get_global_name(env)

        with open(os.path.join(os.getcwd(), 'ray_config', f'{glob_name}_cfg.yaml')) as f:
            config = yaml.safe_load(f)
            np.random.seed(config['seed'])
            del config['seed']

        config['orig_cwd'] = os.getcwd()
        config['env'] = env
        config['glob_name'] = glob_name
        config['device'] = 'cpu'

        # get some hyperparms and remove them from dict
        expert_names = config['local_experts']
        del config['local_experts']
        dict_pos_tol = None
        if 'pos_tol' in config:
            dict_pos_tol = config['pos_tol']
            del config['pos_tol']
        dict_beta = config['beta']
        dict_delta = config['delta']
        dict_phi = config['phi']
        del config['beta']
        del config['delta']
        del config['phi']
        decay_parameter_list = config['decay_parameter']
        del config['decay_parameter']

        for expert in expert_names:

            config['expert'] = expert

            for agent_name in agents:

                # agent name
                glob_agent_name = get_global_agent_name(agent_name)
                config['agent_name'] = agent_name

                # get hyperparameters
                if dict_pos_tol is not None:
                    config['pos_tol'] = dict_pos_tol[agent_name]
                config['beta'] = dict_beta[agent_name]
                config['delta'] = dict_delta[agent_name]
                config['phi'] = dict_phi[agent_name]

                # decay or not
                agent_name_to_show = agent_name
                if agent_name in ['SAC', 'SAG', 'NaiveSAG']:
                    decay_parameter_list = [ False ]

                for decay_parameter in decay_parameter_list:

                    config['decay_parameter'] = decay_parameter
                    if decay_parameter:
                        agent_name_to_show = 'Decreased' + agent_name_to_show
                    else:
                        # to avoid unecessary runs
                        config['delta'] = [ 1 ]

                    # ray preparation
                    hps = [k for k, v in config.items() if type(v) is list]
                    config_ray = config.copy()
                    config_ray = {k: tune.grid_search(v) if type(v) is list else v for k, v in config.items()}
                    config_ray['repeat_run'] = tune.grid_search(list(range(config['repeat_run'])))
                    metric_columns = ['epoch', 'average_return', 'mean_avg_return', 'epoch_time']
                    reporter = tune.CLIReporter(parameter_columns=hps, metric_columns=metric_columns)

                    env_name_folder = env

                    if agent_name in ['SAC']:
                        save_path = f'./ray_results_test/{env_name_folder}/{agent_name_to_show}'
                    else:
                        save_path = f'./ray_results_test/{env_name_folder}/{agent_name_to_show}/{expert}'

                    analysis = run(
                        sac_main,
                        config=config_ray,
                        metric=config_ray['metric'],
                        mode=config_ray['mode'],
                        resources_per_trial={"cpu": 1, "gpu": 1 if config_ray['device'] == 'cuda' else 0},
                        max_concurrent_trials=15,
                        log_to_file=True,
                        local_dir=save_path,
                        trial_name_creator=lambda t: trial_name(t, hps),
                        trial_dirname_creator=lambda t: trial_name(t, hps),
                        progress_reporter=reporter,
                        verbose=1)  # resume=True,
