from collections import defaultdict
import gym 
import os 
from tensorboardX import SummaryWriter

from typing import Tuple, Dict, List, Callable
import itertools
# environments 
from envs.mountain_car.mountain_car import mcar_cfg
from envs.pendula.single_pendulum import  pendulum_cfg
from envs.pendula.double_pendulum import  double_pendulum_cfg
from envs.reacher.reacher import reacher_cfg

#utils
from common.utils import set_overrides, create_path

import logging
import json

def is_safety_gym_env(env_name):
    """
    Checks if the environment is a safety gym environment. 
    :returns: True if the environment is safety gym environment
    """
    return ('Point' in  env_name or 'Car' in env_name or 'Doggo' in env_name) and \
            ('Goal' in  env_name) and \
            ('Static' in  env_name or 'Dynamic' in env_name)

class BaseRunner:
    """Base class for learning the polices."""
    def __init__(
            self,
            experiment_name:str, 
            agent_name:str, 
            task_name:str,
            param_sweep_lists:List[List],
            agent_cfg_overrides:Dict, 
            env_cfg_overrides:Dict        
    ):
        self.experiment_name = experiment_name
        self.agent_name = agent_name
        self.task_name = task_name        
        self.param_sweep_lists = param_sweep_lists
        self.all_overrides = itertools.product(*param_sweep_lists)
        self.agent_cfg_overrides = agent_cfg_overrides
        self.env_cfg_overrides = env_cfg_overrides

    def create_env(
            self,     
            agent_cfg:Dict, 
            env_cfg_override:Dict
    ) -> Tuple[Callable, Callable, Dict]:
        """
        Script for creating environments specified in cofiguration files.
        :param agent_cfg: dictionary with the agent config files 
        :param env_cfg_override: dictionary with ovverides for the environment config files 
        """
        if ('Saute' in self.agent_name or 'Simmer' in self.agent_name) and ('Lagrangian' in self.agent_name or 'CVaR' in self.agent_name):
            env_cfg_override['use_reward_shaping'] = False
        if is_safety_gym_env(agent_cfg['env_name']):
            if  'Static' in  agent_cfg['env_name']:
                from envs.safety_gym.augmented_sg_envs import static_engine_cfg 
                engine_cfg = static_engine_cfg
            elif 'Dynamic' in agent_cfg['env_name']:       
                from envs.safety_gym.augmented_sg_envs import dynamic_engine_cfg 
                engine_cfg = dynamic_engine_cfg
            if 'Point' in  agent_cfg['env_name']:
                engine_cfg['robot_base'] = 'xmls/point.xml'
            elif 'Car' in agent_cfg['env_name']:       
                engine_cfg['robot_base'] = 'xmls/car.xml'
            elif 'Doggo' in agent_cfg['env_name']:       
                engine_cfg['robot_base'] = 'xmls/doggo.xml'
            if 'Goal' in  agent_cfg['env_name']:
                engine_cfg['task'] = 'goal'
            if 'Sauted' in agent_cfg['env_name']:            
                from envs.safety_gym.augmented_sg_envs import AugmentedSafeEngine, saute_env_cfg
                env_cfg = set_overrides(saute_env_cfg, env_cfg_override) 
                engine_cfg['num_steps'] = env_cfg['max_ep_len']
                train_env_fn = lambda: AugmentedSafeEngine(
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="train",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
                             engine_cfg=engine_cfg)
                env_cfg['mode'] = "test"
                test_env_fn = lambda: AugmentedSafeEngine(
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="train",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
                             engine_cfg=engine_cfg)
            else: 
                from envs.safety_gym.augmented_sg_envs import BaselineEngine, baseline_env_cfg
                env_cfg = set_overrides(baseline_env_cfg, env_cfg_override) 
                engine_cfg['num_steps']=env_cfg['max_ep_len']
                train_env_fn = lambda: BaselineEngine(
                             max_ep_len=env_cfg['max_ep_len'],
                             mode="train",
                             engine_cfg=engine_cfg)                
                test_env_fn = lambda: BaselineEngine(                     
                             max_ep_len=env_cfg['max_ep_len'],
                             mode="train", 
                             engine_cfg=engine_cfg)                
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
        elif agent_cfg['env_name'] == 'MountainCar': 
            env_cfg = set_overrides(mcar_cfg, env_cfg_override)                                           
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make(
                "SafeMountainCar-v0", 
                mode="train",
                )
            test_env_fn  = lambda : gym.make(
                "SafeMountainCar-v0", 
                mode="test",                
                )
        elif agent_cfg['env_name'] == 'SautedMountainCar':  
            env_cfg = set_overrides(mcar_cfg, env_cfg_override)  
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SautedMountainCar-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="train",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )
            test_env_fn = lambda : gym.make("SautedMountainCar-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="test",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )
        elif agent_cfg['env_name'] == 'Pendulum': 
            env_cfg = set_overrides(pendulum_cfg, env_cfg_override)                                           
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make(
                "SafePendulum-v0", 
                mode="train",
                )
            test_env_fn  = lambda : gym.make(
                "SafePendulum-v0", 
                mode="test",
                )
        elif agent_cfg['env_name'] == 'SautedPendulum':  
            env_cfg = set_overrides(pendulum_cfg, env_cfg_override)  
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SautedPendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="train",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )
            test_env_fn = lambda : gym.make("SautedPendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="test",
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )
        elif agent_cfg['env_name'] == 'SimmeredPendulum':  
            env_cfg = set_overrides(pendulum_cfg, env_cfg_override)  
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SimmeredPendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="train",
                             max_safety_budget=env_cfg['max_safety_budget'],                             
                             unsafe_reward=env_cfg['unsafe_reward'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )        
            test_env_fn = lambda : gym.make("SimmeredPendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             mode="test",
                             max_safety_budget=env_cfg['max_safety_budget'],
                             unsafe_reward=env_cfg['unsafe_reward'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation'],
            )
        elif agent_cfg['env_name'] == 'DoublePendulum': 
            env_cfg = set_overrides(double_pendulum_cfg, env_cfg_override)       
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SafeDoublePendulum-v0", mode="train")
            test_env_fn  = lambda : gym.make("SafeDoublePendulum-v0", mode="test")
        elif agent_cfg['env_name'] == 'SautedDoublePendulum':    
            env_cfg = set_overrides(double_pendulum_cfg, env_cfg_override)                                        
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SautedDoublePendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             mode="train",
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             unsafe_reward=env_cfg['unsafe_reward'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation']
            )
            test_env_fn = lambda : gym.make("SautedDoublePendulum-v0",                             
                             safety_budget=agent_cfg['safety_budget'], 
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             mode="test",
                             max_ep_len=agent_cfg['max_ep_len'],
                             unsafe_reward=env_cfg['unsafe_reward'],
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation']
            )
        elif agent_cfg['env_name'] == 'Reacher':
            env_cfg = set_overrides(reacher_cfg, env_cfg_override)
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make(
                "SafeReacher-v0",
                mode="train",
                )
            test_env_fn  = lambda : gym.make(
                "SafeReacher-v0",
                mode="test",
                )
        elif agent_cfg['env_name'] == 'SautedReacher':
            env_cfg = set_overrides(reacher_cfg, env_cfg_override)
            agent_cfg['max_ep_len'] =  env_cfg['max_ep_len']
            train_env_fn = lambda : gym.make("SautedReacher-v0",
                             safety_budget=agent_cfg['safety_budget'],
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             unsafe_reward=env_cfg['unsafe_reward'],
                             mode="train",
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation']                             
            )
            test_env_fn = lambda : gym.make("SautedReacher-v0",
                             safety_budget=agent_cfg['safety_budget'],
                             saute_discount_factor=env_cfg['saute_discount_factor'],
                             max_ep_len=agent_cfg['max_ep_len'],
                             unsafe_reward=env_cfg['unsafe_reward'],
                             mode="train",
                             min_rel_budget=env_cfg['min_rel_budget'],
                             max_rel_budget=env_cfg['max_rel_budget'],
                             test_rel_budget=env_cfg['test_rel_budget'],
                             use_reward_shaping=env_cfg['use_reward_shaping'],
                             use_state_augmentation=env_cfg['use_state_augmentation']                           
            )

        else:
            raise NotImplementedError(f"Env {agent_cfg['env_name']} is not implemented")
        return train_env_fn, test_env_fn, agent_cfg, env_cfg

    def set_all_overrides(self):
        """Creating the configrations for all experiments including paths."""
        all_agent_cfg_overrides, all_env_cfg_overrides = {}, {}
        experiment_paths = []
        for count, overrides in enumerate(self.all_overrides):
            cur_agent_overrides = defaultdict(int)
            cur_env_overrides = defaultdict(int)
            cur_params = defaultdict(int)
            for override in overrides:
                if override[0] == 'agent_cfg_overrides':
                    cur_agent_overrides[override[1]] = override[2]
                if override[0] == 'env_cfg_overrides':
                    cur_env_overrides[override[1]] = override[2]
                if override[0] == 'simmer_cfg_overrides':
                    if cur_agent_overrides['simmer_agent_cfg'] == 0:
                        cur_agent_overrides['simmer_agent_cfg'] = defaultdict(int)
                    cur_agent_overrides['simmer_agent_cfg'][override[1]] = override[2]    
                cur_params[override[1]] = override[2]
            all_agent_cfg_overrides[count] = set_overrides(self.agent_cfg_overrides, cur_agent_overrides)                       
            all_env_cfg_overrides[count] = set_overrides(self.env_cfg_overrides, cur_env_overrides)                       
            experiment_paths.append(
                create_path(
                        experiment_name=self.experiment_name, 
                        agent_name=self.agent_name, 
                        task_name=self.task_name,
                        params=cur_params)
                )   
        return all_agent_cfg_overrides, all_env_cfg_overrides, experiment_paths

    def setup_log(self, exp_dir:str, agent_cfg:Dict, env_cfg:Dict) -> Tuple[SummaryWriter, Callable, Callable]:
        """
        Setting the log for the experiment.
        :param exp_dir: string specifying the directory to save experiment data
        :param agent_cfg: dictionary with the agent config files 
        :param env_cfg: dictionary with the environment config files 
        """
        if agent_cfg['log']:
            train_dir = os.path.join(exp_dir, "train")
            if not os.path.isdir(train_dir):
                os.makedirs(train_dir)
            test_dir = os.path.join(exp_dir, 'test')
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)
            writer = SummaryWriter(log_dir=train_dir)
            if agent_cfg['log_updates']:
                logging.basicConfig(level=logging.INFO,
                                    format='%(message)s',
                                    filename=train_dir + '/logs.txt',
                                    filemode='w')
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                log = logging.getLogger()
                log.addHandler(console)
            with open(os.path.join(train_dir, "configurations.json"), 'w') as json_file:
                json.dump(agent_cfg, json_file, sort_keys=False, indent=4) 
                json_file.write(',\n')
                json.dump(env_cfg, json_file, sort_keys=False, indent=4)
        else:
            writer = None
            train_dir = None
            test_dir = None
            if self.agent_cfg['log_updates']:
                logging.basicConfig(format='%(message)s', level=logging.INFO)
                logging.info(agent_cfg)
                logging.info(env_cfg)
        return writer, train_dir, test_dir

