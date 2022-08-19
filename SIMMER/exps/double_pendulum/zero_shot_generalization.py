
from tf_algos.common.runner import TFRunner

def run(smoketest:bool=True):
    """Zero shot generalization to different safety budgtes for the double pendulum Saute TRPO (testing only)."""          
    task_name = 'DoublePendulum' 
    experiment_name = 'zero_shot_generalization' 
    num_exps = 1                                   
    # big overrides 
    agent_cfg_overrides = dict(
        env_name = task_name,   # a necessary override
        discount_factor = 0.99, # a necessary override
        safety_discount_factor = 0.99,
        checkpoint_frequency = 0,
        epochs=301,
    )

    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]      
    test_safety_budgets = [40.0, 60.0, 80.0]         

    max_rel_budgets = [
        ['env_cfg_overrides', 'max_rel_budget', 2.0],
    ] 
    min_rel_budgets = [
        ['env_cfg_overrides', 'min_rel_budget', 0.1],
    ]             
    safety_budgets = [
                ['agent_cfg_overrides', 'safety_budget', 50.0], 
    ]                
    ## training

    param_list = []
    env_cfg_overrides = dict() 
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

    ## testing     
    for test_safety_budget in test_safety_budgets: 
        param_list = []
        env_cfg_overrides = dict(
            test_rel_budget=test_safety_budget / safety_budgets[0][2]
        ) 
        param_list = [safety_budgets,  max_rel_budgets, min_rel_budgets, seeds]    # seeds are the last  
        agent_name = 'SauteTRPO'   # NB!
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 1000000
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
            train=False, 
            test=True, 
            evaluate_last_only=True, # computing only the results only for the last epoch
            data_filename=f"{env_cfg_overrides['test_rel_budget']}_test_results.csv",
            num_exps=num_exps    
        )
        print("done")
    