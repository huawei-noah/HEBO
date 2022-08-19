use_reward_shapings = [
        ['env_cfg_overrides', 'use_reward_shaping', False],
        ['env_cfg_overrides', 'use_reward_shaping', True]
]
use_state_augmentations = [
        ['env_cfg_overrides', 'use_state_augmentation', True],
        ['env_cfg_overrides', 'use_state_augmentation', False]
]
#cfg
cfg = dict(
    experiment_name='ablation_components',
    agents=['SauteTRPO'],
    agent_cfg_overrides=dict(
        env_name="Pendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        safety_discount_factor=0.99,
        checkpoint_frequency=0,
        epochs=200,
    ),
    param_sweep_list=[use_reward_shapings, use_state_augmentations],
    env_cfg_overrides=dict(),
    safety_budgets=[
        ['agent_cfg_overrides', 'safety_budget', 30.0],
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