from tf_algos.common.runner import TFRunner
from typing import List, Dict

def run_saute(        
        experiment_name:str, 
        agents:List[str],
        agent_cfg_overrides:Dict,
        env_cfg_overrides:Dict,
        param_sweep_list:List[List],
        safety_budgets:List[float],
        safety_discount_factors:List[float],
        seeds:List[float],
        train:bool=True,
        test:bool=False,
        data_filename:str="test_results.csv",
        num_exps:int=1, 
        smoketest:bool=False,
        ):
    """
    Run an experiment with Saute/Lagrangian or Vanilla algo 

    :param experiment_name: Name of the experiment  
    :param agents: A list of agents to run, e.g., ['VanillaSAC', 'LagrangianSAC', 'SauteSAC']
    :param agent_cfg_overrides: Overriding parameters for the agent for all experiments
    :param env_cfg_overrides: Overriding parameters of the environment for all experiments
    :param param_sweep_list: A list with a parameter sweep parameters 
    :param safety_budgets: A list with safety budget parameters (not used by Vanilla algos)
    :param safety_discount_factors: A list with discount factor parameters (used only by Lagrangian algos)
    :param seeds: List of seeds
    :param train: Flag for training the policy
    :param test: Flag for testing the policy
    :param data_filename: Filename for saving teh data
    :param num_exps: Number of experiments to run simultaneously 
    :param smoketest: Flag for testing the code
    """

    for agent_name in agents: 
        param_list = param_sweep_list.copy()
        ### vanilla algorithms
        if "Vanilla" in agent_name:
            pass
        ### safe algorithms    
        elif "Saute" in  agent_name:        
            param_list.insert(0, safety_budgets)
        elif "Lagrangian" in agent_name or agent_name == "CPO":    
            param_list.insert(0, safety_discount_factors)
            param_list.insert(0, safety_budgets)
        param_list.append(seeds)           
        ### test            
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 0
            experiment_name = 'test'
            param_list = [[seeds[0]]]
        runner = TFRunner(            
            experiment_name, 
            agent_name, 
            agent_cfg_overrides["env_name"],
            param_sweep_lists=param_list, # seeds are the last  ## policy_lrs, penalty_lrs,
            agent_cfg_overrides=agent_cfg_overrides, 
            env_cfg_overrides=env_cfg_overrides,
        )
        runner.run_experiment(
            train=train, 
            test=test, 
            data_filename=data_filename,
            num_exps=num_exps    
        )
    print("done")
 

def run_simmer(        
        experiment_name:str, 
        agents:List[str],
        agent_cfg_overrides:Dict,
        env_cfg_overrides:Dict,
        param_sweep_list:List[List],
        safety_budgets:List[float],
        safety_discount_factors:List[float],
        seeds:List[float],
        train:bool=True,
        test:bool=False,
        data_filename:str="test_results.csv",
        num_exps:int=1, 
        max_safety_budget:float=120.0,
        smoketest:bool=True,
        ):
    """
    Run an experiment with Simmer/Saute/Lagrangian or Vanilla algo 

    :param experiment_name: Name of the experiment  
    :param agents: A list of agents to run, e.g., ['VanillaSAC', 'LagrangianSAC', 'SauteSAC']
    :param agent_cfg_overrides: Overriding parameters for the agent for all experiments
    :param env_cfg_overrides: Overriding parameters of the environment for all experiments
    :param param_sweep_list: A list with a parameter sweep parameters 
    :param safety_budgets: A list with safety budget parameters (not used by Vanilla algos)
    :param safety_discount_factors: A list with discount factor parameters (used only by Lagrangian algos)
    :param seeds: List of seeds
    :param train: Flag for training the policy
    :param test: Flag for testing the policy
    :param data_filename: Filename for saving teh data
    :param num_exps: Number of experiments to run simultaneously 
    :param smoketest: Flag for testing the code
    :param max_safety_budget: Maximum safety budget for normalization
    """
    

    for agent_name in agents: 
        env_cfg_overrides = dict(max_safety_budget=max_safety_budget) 
        param_list = param_sweep_list  + []
        if "Vanilla" in agent_name:
            pass
        elif "Saute" in  agent_name:        
            param_list.insert(0, safety_budgets)
            if "Lagrangian" in agent_name:
                env_cfg_overrides['saute_discount_factor'] = 0.99
                param_list.insert(1, safety_discount_factors)
        elif "Lagrangian" in agent_name or "CPO" in agent_name:    
            param_list.insert(0, safety_discount_factors)
            param_list.insert(0, safety_budgets)
        param_list.append(seeds)  
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 0
            experiment_name = 'test'
            param_list = [[seeds[0]]]
        runner = TFRunner(            
            experiment_name, 
            agent_name, 
            agent_cfg_overrides["env_name"],
            param_sweep_lists=param_list, # seeds are the last  ## policy_lrs, penalty_lrs,
            agent_cfg_overrides=agent_cfg_overrides, 
            env_cfg_overrides=env_cfg_overrides,
        )
        runner.run_experiment(
            train=train, 
            test=test, 
            data_filename=data_filename,
            num_exps=num_exps    
        )
    print("done")
  