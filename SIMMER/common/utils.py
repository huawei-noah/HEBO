import copy
import os
from typing import Dict
from collections import defaultdict

def set_overrides(cfg:Dict, overrides:Dict):
    """
    Overriding the specified entries in the dictionary.
    :param overrides: override entries
    :param cfg: main dictionary 
    """
    new_cfg = copy.deepcopy(cfg)
    if overrides:
        for key in overrides.keys():
            if type(overrides[key]) is defaultdict:
                if key in new_cfg.keys():
                    new_cfg_kid = new_cfg[key]
                else:
                    new_cfg_kid = dict()
                new_cfg[key] = set_overrides(new_cfg_kid, overrides[key])                
            else:
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