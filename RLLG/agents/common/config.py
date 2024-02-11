from typing import Any, Dict, List, Optional, Tuple


def process_glob_config(config: Dict[str, Any]) \
        -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[float]]:
    """
    Process a global configuration dictionary and extract relevant information.

    Parameters:
    ----------
    config : Dict[str, Any]
        The global configuration dictionary.

    Returns:
    ----------
    Tuple[List[str], Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[float]]
        A tuple containing the extracted information:
        - List of expert names.
        - Dictionary for position tolerance (or None if not present).
        - Dictionary for values for beta depending on the agent.
        - Dictionary for values for delta depending on the agent.
        - Dictionary for values for phi depending on the agent.
        - List of decay parameters.
        """
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
    return expert_names, dict_pos_tol, dict_beta, dict_delta, dict_phi, decay_parameter_list


def process_config_per_agent(config: Dict[str, Any],
                             agent_name: str,
                             dict_beta: Dict[str, Any],
                             dict_delta: Dict[str, Any],
                             dict_phi: Dict[str, Any],
                             dict_pos_tol: Dict[str, Any]) -> None:
    """
    Process the configuration dictionary to make it dependant on the agent

    Parameters:
    ----------
    config : Dict[str, Any]
        The global configuration dictionary to be updated
    dict_pos_tol: Optional[Dict[str, Any]])
        Add pos_tol argument to config dictionary. It is only usefull for the safe cartpole environment

    Returns:
    ----------
    None
        The function does not return anything.
    """
    if dict_pos_tol is not None:
        config['pos_tol'] = dict_pos_tol[agent_name]
    config['beta'] = dict_beta[agent_name]
    config['delta'] = dict_delta[agent_name]
    config['phi'] = dict_phi[agent_name]
