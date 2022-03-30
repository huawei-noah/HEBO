import numpy as np
from safe_rl.utils.load_utils import load_policy
import os
import re
import gym
import pandas as pd
from typing import Tuple, Callable
"""
Scripts for evaluating agents. Copied from safety starter agents with minor modifications.
"""

def evaluate_epoch(
    fpath:str,
    env_fn:gym.Env,
    epoch:int,
    evaluations:int
) -> pd.DataFrame:
    """
    Computes policy evaluations
    :param fpath: path with a saved policy, 
    :param epoch: epoch number for evaluation,
    :param evaluations: number of evaluation episodes.
    """
    # Lists to capture all measurements of performance
    episode_returns = []
    episode_costs = []
    
    # Load environment, policy, and session
    saved_environment, policy, _ = load_policy(fpath=fpath, itr=epoch, deterministic=True)
    if env_fn is None:
        environment = saved_environment
    else:
        environment = env_fn()
    # Run evaluation episodes
    for _ in range(evaluations):
        episode_return, episode_cost = run_policy(environment, policy)
        episode_returns.append(episode_return); episode_costs.append(episode_cost)

    # Present performance metrics
    df = pd.DataFrame({'episode_return':episode_returns, 'episode_cost':episode_costs})
    df['epoch'] = epoch

    return df

# Runs a single episode and records return/cost
def run_policy(
    environment,
    policy
) -> Tuple[np.ndarray, np.ndarray]:
    observation = environment.reset(); done = False
    episode_return, episode_cost = 0, 0
    while not done:
        action = policy(observation)
        action = np.clip(
                        action, 
                        environment.action_space.low, 
                        environment.action_space.high)
        observation, reward, done, info = environment.step(action)
        episode_return += reward
        episode_cost += info.get('cost', 0)

    return episode_return, episode_cost

def evaluate_run(
    path_dir:str,
    env_fn:Callable,
    evaluations: int=100,
    last_only:bool=False # evaluate only the last policy
) -> pd.DataFrame:
    # Figure out how many epochs there are to evaluate
    run_contents = os.listdir(path_dir)
    all_epochs = []
    for epoch in run_contents:
        if epoch.startswith('vars'):
            all_epochs.append(int(re.findall(r'\d+', epoch)[0]))

    df = pd.DataFrame()
    if last_only:
        all_epochs = [max(all_epochs)]
    for idx, epoch in enumerate(all_epochs):
        print(idx/max(all_epochs))
        df = pd.concat([df, evaluate_epoch(fpath=path_dir, env_fn=env_fn, epoch=epoch, evaluations=evaluations)], ignore_index=True)

    return df
