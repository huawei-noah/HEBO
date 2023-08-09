# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.



from copy import deepcopy
import torch
from omegaconf import DictConfig
import os
from ray import tune
import numpy as np

# agents
from agents.algos.sac import SAC
from agents.algos.sag import SAG
from agents.algos.pag import PAG
from agents.algos.pig import PIG
from agents.common.model import TanhGaussianPolicy, ParametrizedPerturbationTanhGaussianPolicy, FullyConnectedQFunction, \
    SamplerPolicy, ExpertSamplerPolicy
from agents.common.replay_buffer import ReplayBuffer, batch_to_torch
from agents.common.sampler import StepSampler, TrajSampler
from agents.common.utils import Timer, set_random_seed, prefix_metrics
from envs.creation import get_env_and_control
from envs.confidence import global_lambda_s

dict_agents = {
    'SAC': SAC,
    'SAG': SAG,
    'PIG': PIG,
    'PAG': PAG,
}


def save_all_models(qf1, qf2, target_qf1, target_qf2, policy, path):
    torch.save(qf1.state_dict(), os.path.join(path, 'qf1'))
    torch.save(qf2.state_dict(), os.path.join(path, 'qf2'))
    torch.save(target_qf1.state_dict(), os.path.join(path, 'target_qf1'))
    torch.save(target_qf2.state_dict(), os.path.join(path, 'target_qf2'))
    torch.save(policy.state_dict(), os.path.join(path, 'policy'))


def load_all_models(qf1, qf2, target_qf1, target_qf2, policy, path):
    qf1.load_state_dict(torch.load(os.path.join(path, 'qf1')))
    qf2.load_state_dict(torch.load(os.path.join(path, 'qf2')))
    target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1')))
    target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2')))
    policy.load_state_dict(torch.load(os.path.join(path, 'policy')))


def main(cfg):
    cfg = DictConfig(cfg)

    # global hyperparameters
    agent_name = cfg['agent_name']
    glob_name = cfg['glob_name']
    num_run = cfg['repeat_run']

    # environment parameters
    limit_cart = None
    reward_end = None
    pos_tol = None
    speed_tol = None
    if 'limit_cart' in cfg:
        limit_cart = cfg['limit_cart']
    if 'reward_end' in cfg:
        reward_end = cfg['reward_end']
    if 'pos_tol' in cfg:
        pos_tol = cfg['pos_tol']
    env_train, local_control_dict_train = get_env_and_control(name=cfg['env'],
                                                              orig_cwd=cfg['orig_cwd'],
                                                              device=cfg['device'],
                                                              limit_cart=limit_cart,
                                                              reward_end=reward_end,
                                                              pos_tol=pos_tol
                                                              )
    env_test, local_control_dict_test = get_env_and_control(name=cfg['env'],
                                                            orig_cwd=cfg['orig_cwd'],
                                                            device=cfg['device'],
                                                            limit_cart=limit_cart,
                                                            reward_end=reward_end,
                                                            pos_tol=pos_tol
                                                            )

    # experts
    expert = cfg['expert']
    lambda_s = global_lambda_s(cfg['glob_name'],
                               expert,
                               device=cfg['device'],
                               pos_tol=pos_tol,
                               speed_tol=speed_tol
                               )
    local_expert = local_control_dict_train[expert]['local_expert']

    train_sampler = StepSampler(env_train, cfg['max_traj_length'])  # .unwrapped
    eval_sampler = TrajSampler(env_test, cfg['max_traj_length'])  # .unwrapped
    replay_buffer = ReplayBuffer(cfg['replay_buffer_size'])
    set_random_seed(cfg["repeat_run"])

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        cfg['network_arch'],
        log_std_multiplier=cfg['policy_log_std_multiplier'],
        log_std_offset=cfg['policy_log_std_offset'],
        activation=cfg['activation_fn']
    )
    sampler_policy = SamplerPolicy(policy, cfg['device'])

    parametrized_perturbation = ParametrizedPerturbationTanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        cfg['network_arch'],
        log_std_multiplier=cfg['policy_log_std_multiplier'],
        log_std_offset=cfg['policy_log_std_offset'],
        activation=cfg['activation_fn'],
        phi=cfg['phi']
    )
    sampler_parametrized_perturbation = ExpertSamplerPolicy(parametrized_perturbation, cfg['device'])

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        cfg['network_arch'],
        activation=cfg['activation_fn']
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        cfg['network_arch'],
        activation=cfg['activation_fn']
    )
    target_qf2 = deepcopy(qf2)

    if cfg['target_entropy'] >= 0.0:
        cfg['target_entropy'] = -np.prod(eval_sampler.env.action_space.shape).item()

    # Get agent
    if cfg['agent_name'] == 'SAC':
        agent = dict_agents[agent_name](cfg,
                                        policy,
                                        sampler_policy,
                                        qf1,
                                        qf2,
                                        target_qf1,
                                        target_qf2)
    elif cfg['agent_name'] == 'SAG':
        agent = dict_agents[agent_name](cfg,
                                        policy,
                                        sampler_policy,
                                        qf1,
                                        qf2,
                                        target_qf1,
                                        target_qf2,
                                        use_local=lambda_s,
                                        local_expert=local_expert)
    elif cfg['agent_name'] == 'PIG':
        agent = dict_agents[agent_name](cfg,
                                        policy,
                                        sampler_policy,
                                        qf1,
                                        qf2,
                                        target_qf1,
                                        target_qf2,
                                        use_local=lambda_s,
                                        local_expert=local_expert,
                                        beta=cfg['beta'])
    else:
        agent = dict_agents[agent_name](cfg,
                                        policy,
                                        sampler_policy,
                                        qf1,
                                        qf2,
                                        target_qf1,
                                        target_qf2,
                                        use_local=lambda_s,
                                        local_expert=local_expert,
                                        parametrized_perturbation=parametrized_perturbation,
                                        sampler_parametrized_perturbation=sampler_parametrized_perturbation)
    agent.torch_to_device(cfg['device'])

    # put beta right if PAG without decay parameter
    if not cfg['decay_parameter'] and agent_name == 'PAG':
        agent.beta = 0

    # Sample data initially
    with Timer() as initial_rollout_timer:
        train_sampler.sample(
            agent,
            cfg['n_initial_env_steps'],
            deterministic=False,
            replay_buffer=replay_buffer
        )

    mean_avg_return = []
    for epoch in range(cfg['n_epochs']):

        # decrease norm scale if necessary
        if cfg['decay_parameter'] and epoch % 50 == 0 and epoch > 0:
            if agent_name in ['PIG', 'PAG']:
                agent.beta *= cfg['decay_rate']

        metrics = {}

        # Sample data
        with Timer() as rollout_timer:
            train_sampler.sample(
                agent,
                cfg['n_env_steps_per_epoch'],
                deterministic=False,
                replay_buffer=replay_buffer
            )
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        # Training
        with Timer() as train_timer:
            for batch_idx in range(cfg['n_train_step_per_epoch']):
                batch = batch_to_torch(replay_buffer.sample(cfg['batch_size']), cfg['device'])
                if batch_idx + 1 == cfg['n_train_step_per_epoch']:
                    metrics.update(prefix_metrics(agent.train(batch), cfg['agent_name']))
                else:
                    agent.train(batch)

        # Evaluation
        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % cfg['eval_period'] == 0 or epoch == cfg['n_epochs']-1:
                trajs = eval_sampler.sample(
                    agent, cfg['eval_n_trajs'], deterministic=True
                )
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if epoch // cfg['eval_period'] < 10:
                    mean_avg_return.append(metrics['average_return'])
                else:
                    mean_avg_return[(epoch // cfg['eval_period']) % 10] = metrics['average_return']
                metrics['mean_avg_return'] = np.mean(mean_avg_return)

                # save number of times we the different policies in avg and variance
                metrics[f'mean_play_local'] = np.mean(
                    [np.sum(np.array(t[f'list_use_local_current'], dtype=bool).astype(int)) for t in trajs])
                metrics[f'std_play_local'] = np.std(
                    [np.sum(np.array(t[f'list_use_local_current'], dtype=bool).astype(int)) for t in trajs])
                metrics[f'failures'] = np.mean([np.sum(t[f'failures']) for t in trajs])

                if agent_name in ['PIG', 'PAG']:
                    metrics[f'beta'] = agent.beta

        if epoch == 0 or (epoch + 1) % cfg['eval_period'] == 0 or epoch == cfg['n_epochs'] - 1:
            metrics['epoch'] = epoch
            metrics['rollout_time'] = rollout_timer()
            metrics['train_time'] = train_timer()
            metrics['eval_time'] = eval_timer()
            metrics['epoch_time'] = train_timer() + eval_timer()

            # Report metrics
            tune.report(**metrics)

        if epoch % cfg['num_epoch_save'] == 0 and cfg['agent_name'] == 'SAC' and epoch > 0:
            act_fn = cfg['activation_fn']
            save_path_init = os.path.join(cfg['orig_cwd'],
                                          'envs',
                                          glob_name,
                                          'models')
            os.makedirs(save_path_init, exist_ok=True)
            save_path = os.path.join(save_path_init,
                                     f'training_policy_sac_act_{act_fn}_{epoch}_{num_run}')
            torch.save(agent.policy.state_dict(), save_path)

    # save expert SAC model
    # path = os.path.join(cfg['orig_cwd'], 'envs', cfg['env'], 'models')
    # os.makedirs(path, exist_ok=True)
    # save_all_models(qf1, qf2, target_qf1, target_qf2, policy, path)
