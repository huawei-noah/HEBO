
from tf_algos.common.runner import TFRunner

def run_tf_trpo_double_pendulum_v2(        
        experiment_name:str=None, 
        num_exps:int=1, 
        smoketest:bool=True
        ):
    """
    Tuning Lagrangian TRPO.
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

    safety_budgets = [
                # ['agent_cfg_overrides', 'safety_budget', 20.0], 
                ['agent_cfg_overrides', 'safety_budget', 40.0], 
                # ['agent_cfg_overrides', 'safety_budget', 60.0], 
                # ['agent_cfg_overrides', 'safety_budget', 80.0], 
    ]    
    penalty_lrs = [
        # ['agent_cfg_overrides', 'penalty_lr', 1e-3], 
        # ['agent_cfg_overrides', 'penalty_lr', 1e-2], 
        ['agent_cfg_overrides', 'penalty_lr', 5e-2], 
    ]    
    value_fn_lrs = [
        ['agent_cfg_overrides', 'value_fn_lr', 5e-3],
        # ['agent_cfg_overrides', 'value_fn_lr', 1e-3],
        # ['agent_cfg_overrides', 'value_fn_lr', 5e-4],
    ]   
    backtrack_iterss = [
        ['agent_cfg_overrides', 'backtrack_iters', 20],
        # ['agent_cfg_overrides', 'backtrack_iters', 15],
        # ['agent_cfg_overrides', 'backtrack_iters', 10],
    ] 
    steps_per_epochs = [
        # ['agent_cfg_overrides', 'steps_per_epoch', 20000],
        # ['agent_cfg_overrides', 'steps_per_epoch', 10000],
        ['agent_cfg_overrides', 'steps_per_epoch', 4000]
    ] 
    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]      

    for agent_name in ['LagrangianTRPO']: 
        safety_discount_factors = [
            ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
        ] 
        env_cfg_overrides = {}
        param_list = []
        if agent_name == 'VanillaTRPO':
            param_list = [seeds]     
        if agent_name == 'LagrangianTRPO':    
            param_list = [safety_budgets, safety_discount_factors, steps_per_epochs, backtrack_iterss, penalty_lrs, value_fn_lrs, seeds]
        if agent_name == 'CPO':
            param_list = [safety_budgets, safety_discount_factors, seeds]    
        if agent_name == 'SauteTRPO':
            max_rel_budgets = [
                ['env_cfg_overrides', 'max_rel_budget', 2.0],
            ] 
            min_rel_budgets = [
                ['env_cfg_overrides', 'min_rel_budget', 0.1],
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
            param_list = [safety_budgets,  max_rel_budgets, min_rel_budgets, seeds]                     
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
            param_sweep_lists=param_list, # seeds are the last  ## policy_lrs, penalty_lrs,
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
    