import numpy as np

max_safety_budget=35.
n_epochs=801    
min_safety_budget=15.0
n_states=6                
state_space=list(np.linspace(min_safety_budget, max_safety_budget, n_states))
global_action_space=[min_safety_budget, max_safety_budget]
 
simmer_polyak = [
    # ['simmer_cfg_overrides', 'polyak_update', 0.0005],
    ['simmer_cfg_overrides', 'polyak_update', 0.001],
    ['simmer_cfg_overrides', 'polyak_update', 0.005],
    ['simmer_cfg_overrides', 'polyak_update', 0.99],
    ['simmer_cfg_overrides', 'polyak_update', 0.995],
    ['simmer_cfg_overrides', 'polyak_update', 1.0],
]      
qsimmerR = [
    ['simmer_cfg_overrides', 'reward_thresh', .1],
    ['simmer_cfg_overrides', 'reward_thresh', 1],
    ['simmer_cfg_overrides', 'reward_thresh', 5],
]      
qsimmerlr = [
    # ['simmer_cfg_overrides', 'lr', .005],
    ['simmer_cfg_overrides', 'lr', 0.01],
    ['simmer_cfg_overrides', 'lr', 0.05],
    ['simmer_cfg_overrides', 'lr', 0.1],
    ['simmer_cfg_overrides', 'lr', 0.5],
]   
simmer_q_params_cfg = dict(
    type="q", 
    lr=.01, 
    simmer_gamma=0.99, 
    epsilon_greedy=0.95, 
    polyak_update=0.01,
    reward_thresh=2,
    action_penalty=0.01,
    state_space=state_space,
    action_space=[-1, 0, 1],
    trial_length=n_epochs,
    global_action_space=global_action_space
)         

#cfg
cfg = dict(
    experiment_name='simmer/q',
    agents=['SimmerSautePPO'],
    agent_cfg_overrides=dict(
        env_name="Pendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        checkpoint_frequency=0,
        n_test_episodes=100,
        penalty_lr=3e-2,
        epochs=n_epochs,
        simmer_agent_cfg=simmer_q_params_cfg
    ),
    env_cfg_overrides=dict(),
    param_sweep_list=[qsimmerR, qsimmerlr, simmer_polyak],
    safety_budgets=[
        ['agent_cfg_overrides', 'safety_budget', max_safety_budget],
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
    num_exps=3, 
    max_safety_budget=max_safety_budget,
)