#cfg
cfg = dict(
    experiment_name='performance',
    agents=['CPO', 'SauteTRPO', 'VanillaTRPO'],
    agent_cfg_overrides=dict(
        env_name="DoublePendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        safety_discount_factor=0.99,
        checkpoint_frequency=0,
        epochs=301,
    ),
    env_cfg_overrides=dict(),
    param_sweep_list=[],
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
    num_exps=5, 
)