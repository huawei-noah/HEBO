
from tf_algos.common.runner import TFRunner

def run_tf_trpo_double_pendulum_v1(        
        experiment_name:str=None, 
        num_exps:int=1, 
        smoketest:bool=True
        ):
    """
    Running Lagrangian TRPO, CPO, Vanilla TRPO, Saute TRPO on the double pendulum environment.
    """          
    if experiment_name is None:
        experiment_name = 'performance'         
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
    for agent_name in ['SauteTRPO', 'CPO', 'VanillaTRPO']: # Lagrangian TRPO is not run in this file
        safety_discount_factors = [
            ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
        ] 
        safety_budgets = [
                    ['agent_cfg_overrides', 'safety_budget', 40.0], 
                    # ['agent_cfg_overrides', 'safety_budget', 20.0], 
                    # ['agent_cfg_overrides', 'safety_budget', 60.0], 
                    # ['agent_cfg_overrides', 'safety_budget', 80.0], 
        ]                

        env_cfg_overrides = {}
        param_list = []
        if agent_name == 'VanillaTRPO':
            param_list = [seeds]     
        if agent_name == 'LagrangianTRPO' or agent_name == 'SauteLagrangianTRPO':    
            param_list = [safety_budgets, safety_discount_factors,  seeds] 
        if agent_name == 'CPO':
            param_list = [safety_budgets, safety_discount_factors, seeds]    
        if agent_name == 'SauteTRPO':
            max_rel_budgets = [
                ['env_cfg_overrides', 'max_rel_budget', 1.0],
            ] 
            min_rel_budgets = [
                ['env_cfg_overrides', 'min_rel_budget', 1.0],
            ]             
            param_list = [safety_budgets,  max_rel_budgets, min_rel_budgets, seeds]                 
        if agent_name == 'SauteTRPO_allbudgets':    
            env_cfg_overrides = dict(
                test_rel_budget=1.0,
            ) 

            max_rel_budgets = [
                ['env_cfg_overrides', 'max_rel_budget', 2.0],
            ] 
            min_rel_budgets = [
                ['env_cfg_overrides', 'min_rel_budget', 0.1],
            ]             
            safety_budgets = [
                        ['agent_cfg_overrides', 'safety_budget', 50.0], 
            ]                
            param_list = [safety_budgets,  max_rel_budgets, min_rel_budgets, seeds]    # seeds are the last 
            agent_name = 'SauteTRPO'   # NB!
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
    