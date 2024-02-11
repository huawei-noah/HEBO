# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from typing import List
import os
import numpy as np
import yaml
from ray import tune
from ray.tune import run, Experiment

from agents.common.utils import get_global_name, get_global_agent_name
from agents.common.config import process_glob_config, process_config_per_agent
from sac_main_fn import main as sac_main

os.environ["Timer"] = '1'


def trial_name(trial: Experiment, hp_to_write: List[str]) -> str:
    """
    Generate a unique identifier for a trial based on specified hyperparameters and trial information.

    Parameters:
    ----------
    trial : ray.tune.Experiment
        The Ray Tune Experiment for which to generate the identifier.
    hp_to_write : List[str]
        List of hyperparameter names to include in the identifier.

    Returns:
    ----------
    str
        The generated trial identifier.
    """
    ti = 'repeat_run'
    identifier = ','.join([f'{hp}={trial.config[hp]}' for hp in hp_to_write]) + \
                 f',trial={trial.config[ti]},id={trial.trial_id}'
    return identifier


if __name__ == '__main__':

    envs = ['ball_in_cup']
    agents = [
        'SAC',
        'SAG',
        'PIG',
        'PAG',
    ]

    for env in envs:
        # get global name to retrieve configs
        glob_name = get_global_name(env)

        # retrieve config
        with open(os.path.join(os.getcwd(), 'ray_config', f'{glob_name}_cfg.yaml')) as f:
            config = yaml.safe_load(f)
            np.random.seed(config['seed'])
            del config['seed']

        # add important elements to the config file
        config['orig_cwd'] = os.getcwd()
        config['env'] = env
        config['glob_name'] = glob_name
        config['device'] = 'cpu'

        # process config and retrieve elements for loops
        expert_names, dict_pos_tol, dict_beta, dict_delta, dict_phi, decay_parameter_list = process_glob_config(config)

        # loop over experts (if multiple experts)
        for expert in expert_names:

            config['expert'] = expert

            # loop over agents (if multiple agents)
            for agent_name in agents:

                # agent name
                glob_agent_name = get_global_agent_name(agent_name)
                config['agent_name'] = agent_name

                # further process hyperparamers to make them dependent on agent
                process_config_per_agent(config, agent_name, dict_beta, dict_delta, dict_phi, dict_pos_tol)

                # decay or not (only relevant for PAG)
                agent_name_to_show = agent_name
                if agent_name in ['SAC', 'SAG', 'NaiveSAG']:
                    decay_parameter_list = [False]

                for decay_parameter in decay_parameter_list:

                    # to avoid unecessary runs
                    config['decay_parameter'] = decay_parameter
                    if decay_parameter:
                        agent_name_to_show = 'Decreased' + agent_name_to_show
                    else:
                        # to avoid unecessary runs
                        config['delta'] = [1]

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
