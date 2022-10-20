#cfg
cfg = dict(
    experiment_name='plug_n_play',
    agents=['SauteSAC'],
    agent_cfg_overrides=dict(
        env_name="Pendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        steps_per_epoch=200,
        epochs=200,
        checkpoint_frequency=0,
        penalty_lr=5e-2,
        n_test_episodes=100,
    ),
    env_cfg_overrides=dict(),
    param_sweep_list=[],
    safety_budgets=[
        ['agent_cfg_overrides', 'safety_budget', 10.0],
        ['agent_cfg_overrides', 'safety_budget', 20.0],
        ['agent_cfg_overrides', 'safety_budget', 30.0],
        ['agent_cfg_overrides', 'safety_budget', 40.0],
        ['agent_cfg_overrides', 'safety_budget', 50.0],
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

