#cfg
cfg = dict(
    experiment_name='performance',
    agents=['LagrangianTRPO'],
    agent_cfg_overrides=dict(
        env_name="DoublePendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        safety_discount_factor=0.99,
        checkpoint_frequency=0,
        penalty_lr=5e-2,
        value_fn_lr=5e-3,
        backtrack_iters=20,
        steps_per_epoch=4000,        
        epochs=301,
    ),
    param_sweep_list=[],
    env_cfg_overrides=dict(),
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