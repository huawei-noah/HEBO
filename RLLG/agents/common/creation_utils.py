from typing import Type, Any, Dict, List, Optional, Tuple
from envs.creation import get_env_and_control
from agents.algos.sac import SAC
from agents.algos.sag import SAG
from agents.algos.pag import PAG
from agents.algos.pig import PIG
from agents.common.model import SamplerPolicy, ExpertSamplerPolicy
from envs.cartpole.confidence import LambdaS
import gym
import torch


dict_agents = {
    'SAC': SAC,
    'SAG': SAG,
    'PIG': PIG,
    'PAG': PAG,
}


def create_envs(cfg: Dict[str, Any]) -> Tuple[Type[gym.Env], Dict[str, Any], Type[gym.Env], Dict[str, Any]]:
    """
    Create training and testing environments with associated local control dictionaries based on the provided configuration.

    Parameters:
    ----------
    cfg : Dict[str, Any]
        The configuration dictionary

    Returns:
    ----------
    Tuple[Type[gym.Env], Dict[str, Any], Type[gym.Env], Dict[str, Any]]
        A tuple containing the training and testing environments along with their respective local control dictionaries.
    """
    limit_cart = None
    reward_end = None
    pos_tol = None
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
    return env_train, local_control_dict_train, env_test, local_control_dict_test


def create_agent(cfg: Dict[str, Any],
                 agent_name: str,
                 policy: torch.nn.Module,
                 sampler_policy: SamplerPolicy,
                 qf1: torch.nn.Module,
                 qf2: torch.nn.Module,
                 target_qf1: torch.nn.Module,
                 target_qf2: torch.nn.Module,
                 lambda_s: Optional[LambdaS] = None,
                 local_expert: Optional[Any] = None,
                 parametrized_perturbation: Optional[Type[torch.nn.Module]] = None,
                 sampler_parametrized_perturbation: Optional[Type[ExpertSamplerPolicy]] = None) \
        -> Any:
    """
    Create an instance of an RL agent based on the specified configuration and components.

    Parameters:
    ----------
    cfg : Dict[str, Any]
        The configuration dictionary.
    agent_name : str
        The name of the agent to be created.
    policy : Type[torch.nn.Module]
        The policy network.
    sampler_policy : Type[SamplerPolicy]
        The policy sampler.
    qf1 : Type[torch.nn.Module]
        The first critic network.
    qf2 : Type[torch.nn.Module]
        The second critic network.
    target_qf1 : Type[torch.nn.Module]
        The target network for the first critic.
    target_qf2 : Type[torch.nn.Module]
        The target network for the second critic.
    lambda_s : Optional[Type[LambdaS]]
        The lambda_s confidence class (optional).
    local_expert : Optional[Type[Any]]
        The local expert (optional, and can be under any form).
    parametrized_perturbation : Optional[Type[torch.nn.Module]]
        The parametrized perturbation network (optional).
    sampler_parametrized_perturbation : Optional[Type[ExpertSamplerPolicy]]
        The sampler for the parametrized perturbation network (optional).

    Returns:
    ----------
    Any
        An instance of the specified RL agent.
    """
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
    return agent
