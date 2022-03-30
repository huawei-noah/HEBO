
from tf_algos.common.runner import TFRunner

def run_tf_ppo_pendulum_v1(        
        experiment_name:str=None, 
        num_exps:int=1, 
        smoketest:bool=True
        ):
    """
    Runs PPO algorithms for safety: 
    VanillaPPO    -  Vanilla PPO,
    LagrangianPPO  -  PPO with a lagrangian constraint,
    SautePPO       -  Saute PPO,  
    SauteLangrangianPPO -  Lagrangian PPO with safety state augmentation.
    """

    if experiment_name is None:
        experiment_name = 'plug_n_play'        
    task_name = 'Pendulum'                                    
    # big overrides 
    agent_cfg_overrides = dict(
        env_name = task_name,   # a necessary override
        discount_factor = 0.99, # a necessary override
        checkpoint_frequency = 0,
        n_test_episodes=100,
        penalty_lr=5e-2,
        epochs=200
    )

    env_cfg_overrides = {}

    safety_budgets = [
        ['agent_cfg_overrides', 'safety_budget', 30.0],
    ]    
 
    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]      

    for agent_name in ['VanillaPPO', 'SautePPO', 'LagrangianPPO']: 
        safety_discount_factors = [
            ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
        ] 
        env_cfg_overrides = {}
        param_list = []
        if agent_name == 'VanillaPPO':
            param_list = [seeds]     
        if agent_name == 'LagrangianPPO':    
            param_list = [safety_budgets, safety_discount_factors, seeds]
        if agent_name == 'SautePPO':        
            param_list = [safety_budgets, seeds]   
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
 