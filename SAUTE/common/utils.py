import copy
import os
from typing import Dict


def set_overrides(cfg:Dict, overrides:Dict):
    """
    Overriding the specified entries in the dictionary.
    :param overrides: override entries
    :param cfg: main dictionary 
    """
    new_cfg = copy.deepcopy(cfg)
    if overrides:
        for key in overrides.keys():
            new_cfg[key] = overrides[key]
    return new_cfg    

def create_path(experiment_name:str, agent_name:str, task_name:str, params:Dict):
    """Create a path for saving the experiments."""
    exp_dir = os.path.join('logs',                            
                           experiment_name,
                           task_name,
                           agent_name)
    for key in params.keys():
        exp_dir = os.path.join(exp_dir, key + '_' + str(params[key]))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir  