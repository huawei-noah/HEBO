from  tf_algos.simmer_agents.pid_simmer_agent import reference_generator
import numpy as np

max_safety_budget=35.
n_epochs = 801    
min_safety_budget = 20.0
n_states = 6                
max_action = 2 * (max_safety_budget - min_safety_budget) / float(n_states)
state_space = list(np.linspace(min_safety_budget, max_safety_budget, n_states))
global_action_space = [min_safety_budget-5.0, max_safety_budget]
 
simmerK = [
    ['simmer_cfg_overrides', 'K', .0],
    ['simmer_cfg_overrides', 'K', .005],
    ['simmer_cfg_overrides', 'K', .01],
]      
simmerKaw = [
    ['simmer_cfg_overrides', 'Kaw', .0],
    ['simmer_cfg_overrides', 'Kaw', .01],
    ['simmer_cfg_overrides', 'Kaw', .04],
]      
simmerKi = [
    ['simmer_cfg_overrides', 'Ki', .005],
    ['simmer_cfg_overrides', 'Ki', .01],
    ['simmer_cfg_overrides', 'Ki', .04],
]      
simmer_polyak = [
    ['simmer_cfg_overrides', 'polyak_update', 0.001],
    ['simmer_cfg_overrides', 'polyak_update', 0.005],
    ['simmer_cfg_overrides', 'polyak_update', 0.99],
    ['simmer_cfg_overrides', 'polyak_update', 0.995],
    ['simmer_cfg_overrides', 'polyak_update', 1.0],
] 

simmer_pid_params_cfg = dict(
    type="pid", 
    K=0, 
    Ki=0, 
    Kaw=1, 
    polyak_update=1.0,
    state_space=state_space,
    action_space=[-max_action, max_action],
    trial_length=n_epochs,
    global_action_space=global_action_space
) 
simmer_pid_params_cfg['refs'] = reference_generator(
    ref_space=simmer_pid_params_cfg['state_space'], 
    n_epochs=n_epochs, 
    ref_type="increase"
)
#cfg
cfg = dict(
    experiment_name='simmer/pi',
    agents=['SimmerSautePPO'],
    agent_cfg_overrides=dict(
        env_name="Pendulum",   # a necessary override
        discount_factor=0.99, # a necessary override
        checkpoint_frequency=0,
        n_test_episodes=100,
        penalty_lr=5e-2,
        epochs=n_epochs,
        simmer_agent_cfg=simmer_pid_params_cfg
    ),
    env_cfg_overrides=dict(),
    param_sweep_list=[simmerK, simmerKi, simmerKaw, simmer_polyak],
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