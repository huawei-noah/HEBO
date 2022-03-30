from tf_algos.common.runner import TFRunner


def run_tf_trpo_safety_gym_v1(
        experiment_name: str = None,
        num_exps: int = 1,
        smoketest: bool = True
):
    """Running experiments for safety gym environments."""
    if experiment_name is None:
        experiment_name = 'performance'
    task_name = 'StaticPointGoal'  # 'StaticCarGoal' 
    # big overrides
    agent_cfg_overrides = dict(
        env_name=task_name,  # a necessary override
        discount_factor=0.99,  # a necessary override
        safety_discount_factor=0.99,
        checkpoint_frequency=0,
        epochs=500,
    )

    env_cfg_overrides = dict(
    )
    # parameter sweep
    safety_budgets = [
        ['agent_cfg_overrides', 'safety_budget', 10.0],
    ]
    penalty_lrs = [
        ['agent_cfg_overrides', 'penalty_lr', 3e-2],
    ]
    value_fn_lrs = [
        ['agent_cfg_overrides', 'value_fn_lr', 5e-3],
    ]
    steps_per_epochs = [
        ['agent_cfg_overrides', 'steps_per_epoch', 10000],
    ]
    seeds = [
        ['agent_cfg_overrides', 'seed', 42],
        ['agent_cfg_overrides', 'seed', 4242],
        ['agent_cfg_overrides', 'seed', 424242],
        ['agent_cfg_overrides', 'seed', 42424242],
        ['agent_cfg_overrides', 'seed', 4242424242],
    ]
    for agent_name in ['VanillaTRPO',  'LagrangianTRPO', 'SauteTRPO', "CPO"]: 
        safety_discount_factors = [
            ['agent_cfg_overrides', 'safety_discount_factor', 0.99],
        ]
        env_cfg_overrides = {}
        param_list = []
        if agent_name == 'VanillaTRPO':
            param_list = [seeds]
        if agent_name == 'LagrangianTRPO':
            param_list = [safety_budgets, safety_discount_factors, penalty_lrs, value_fn_lrs, seeds]
        if agent_name == 'CPO':
            param_list = [safety_budgets, safety_discount_factors, seeds]
        if agent_name == 'SauteTRPO':
            param_list = [safety_budgets, steps_per_epochs, seeds]
        if smoketest:
            agent_cfg_overrides['epochs'] = 2
            agent_cfg_overrides['checkpoint_frequency'] = 0
            experiment_name = 'test'
            param_list = [[seeds[0]]]
        runner = TFRunner(
            experiment_name,
            agent_name,
            task_name,
            param_sweep_lists=param_list,  # seeds are the last 
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
