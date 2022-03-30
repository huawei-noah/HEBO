from tf_algos.common.runner import TFRunner
"""
Runs SAC algorithms for safety: 
VanillaSAC     -  Vanilla SAC,
LagrangianSAC  -  SAC with a lagrangian constraint,
SauteSAC       -  Saute SAC,  
WorstCaseSAC   -  distributional SAC with a c-var constraint  https://www.st.ewi.tudelft.nl/mtjspaan/pub/Yang21aaai.pdf .
"""
def run_tf_sac_v1(        
        experiment_name:str=None, 
        num_exps:int=1, 
        smoketest:bool=True
        ):
    if experiment_name is None:
        experiment_name = 'test2/example'        
    task_name = 'MountainCar'                                    
    # big overrides 
    agent_cfg_overrides = dict(
        env_name = task_name,   # a necessary override
        discount_factor = 0.99, # a necessary override
        steps_per_epoch = 999,
        epochs=1000,
        checkpoint_frequency = 0,
        penalty_lr=5e-2,
        n_test_episodes=100,
    )

    # parameter sweep 
    safety_budgets = [
        ['agent_cfg_overrides', 'safety_budget', 50.0],
    ]    

    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]        

    # 'SauteLagrangianSAC',  'VanillaSAC', 'LagrangianSAC',
    for agent_name in [ 'SauteSAC']: 
        safety_discount_factors = [
            ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
        ] 
        env_cfg_overrides = {}
        if agent_name == 'VanillaSAC':
            param_list = [seeds]     
        if agent_name == 'WorstCaseSAC':
            param_list = [safety_budgets, safety_discount_factors, seeds]
        if agent_name == 'LagrangianSAC' or agent_name == 'SauteLagrangianSAC':    
            param_list = [safety_budgets, safety_discount_factors, seeds]
        if agent_name == 'SauteSAC':       
            param_list = [safety_budgets,  seeds]   
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 1000000
            experiment_name = 'test'
            param_list = [[seeds[0]]]
        runner = TFRunner(            
                experiment_name, 
                agent_name, 
                task_name,
                param_sweep_lists=param_list, # seeds are the last
                agent_cfg_overrides=agent_cfg_overrides, 
                env_cfg_overrides=env_cfg_overrides,
        )
        runner.run_experiment(
                train=True, 
                test=False, 
                # evaluate_last_only=True,
                data_filename="test_results.csv",
                num_exps=num_exps    
        )
    print("done")
 