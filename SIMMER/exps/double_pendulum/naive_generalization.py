
from tf_algos.common.runner import TFRunner

def run(smoketest:bool=True):
    """Naive generalization to different safety budgtes for the double pendulum Saute TRPO (testing only)."""          
    experiment_name = 'performance' # we only evaluate here already trained policies
    task_name = 'DoublePendulum'                                    
    num_exps = 1 
    # big overrides 
    agent_cfg_overrides = dict(
        env_name = task_name,   # a necessary override
        discount_factor = 0.99, # a necessary override
        safety_discount_factor = 1.0,
        checkpoint_frequency = 0,
        epochs=301,
    )
    env_cfg_overrides = dict()
    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]      
    test_safety_budgets = [40.0, 80.0]         
    safety_budgets = [
                ['agent_cfg_overrides', 'safety_budget', 60.0], 
    ]                

    for test_safety_budget in test_safety_budgets: 
        param_list = []
        env_cfg_overrides['test_rel_budget'] = test_safety_budget / safety_budgets[0][2]
        param_list = [safety_budgets, seeds]    # seeds are the last  ## policy_lrs, penalty_lrs,                 
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
            train=False, # no training is needed, this flag should be false
            test=True, 
            evaluate_last_only=True, # computing only the results only for the last epoch
            data_filename=f"{test_safety_budget}_test_results.csv",
            num_exps=num_exps    
        )
        print("done")
    