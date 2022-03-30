
from tf_algos.common.runner import TFRunner

def run_tf_trpo_double_pendulum_v4(        
        experiment_name:str=None, 
        num_exps:int=1, 
        smoketest:bool=True
        ):
    """"Ablation over unsafe reward value for the double pendulum Saute TRPO."""
    if experiment_name is None:
        experiment_name = 'ablation/unsafe_val'         
    task_name = 'DoublePendulum'                                    
    # big overrides 
    agent_cfg_overrides = dict(
        env_name = task_name,   # a necessary override
        discount_factor = 0.99, # a necessary override
        safety_discount_factor = 0.99,
        checkpoint_frequency = 0,
        epochs=300,
    )

    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]      
    safety_budgets = [
                ['agent_cfg_overrides', 'safety_budget', 40.0], 
    ]                
    unsafe_rewards = [
        ['env_cfg_overrides', 'unsafe_reward', -0.0], 
        ['env_cfg_overrides', 'unsafe_reward', -10.0], 
        ['env_cfg_overrides', 'unsafe_reward', -100.0], 
        ['env_cfg_overrides', 'unsafe_reward', -1000.0], 
        ['env_cfg_overrides', 'unsafe_reward', -10000.0],         
        ['env_cfg_overrides', 'unsafe_reward', -100000.0]         
    ]
    for agent_name in ['SauteTRPO']:
        env_cfg_overrides = {}
        param_list = []
        if agent_name == 'SauteTRPO':
            param_list = [safety_budgets,  unsafe_rewards, seeds]                 
        else:
            raise NotImplementedError
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 0
            experiment_name = 'test'
            param_list = [[seeds[0]]]
        runner = TFRunner(            
            experiment_name, 
            agent_name, 
            task_name,
            param_sweep_lists=param_list, 
            agent_cfg_overrides=agent_cfg_overrides, 
            env_cfg_overrides=env_cfg_overrides,
        )
        runner.run_experiment(
            train=True, 
            test=False, 
            data_filename="test_results.csv",
            num_exps=num_exps    
        )
        print("done")
    