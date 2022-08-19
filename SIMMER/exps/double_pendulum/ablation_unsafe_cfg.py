unsafe_rewards = [
        ['env_cfg_overrides', 'unsafe_reward', -0.0], 
        ['env_cfg_overrides', 'unsafe_reward', -10.0], 
        ['env_cfg_overrides', 'unsafe_reward', -100.0], 
        ['env_cfg_overrides', 'unsafe_reward', -1000.0], 
        ['env_cfg_overrides', 'unsafe_reward', -10000.0],         
        ['env_cfg_overrides', 'unsafe_reward', -100000.0]         
    ]
#cfg
cfg = dict(
    experiment_name='ablation_unsafe_val',
    agents=['SauteTRPO'],
    agent_cfg_overrides=dict(
        env_name="DoublePendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        safety_discount_factor=0.99,
        checkpoint_frequency=0,
        epochs=301,
    ),
    env_cfg_overrides=dict(),
    param_sweep_list=[unsafe_rewards],
    safety_budgets=[
        ['agent_cfg_overrides', 'safety_budget', 40.0],
    ],    
    safety_discount_factors=[
        ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
    ],
    seeds=[
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ],
    train=True,
    test=False,
    data_filename="test_results.csv",
    num_exps=6, 
)