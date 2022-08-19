import sys
import numpy as np

sys.path.append(".")

if __name__ == "__main__":
    max_safety_budget = 30.0
    min_safety_budget = 10.0
    n_states = 4    
    state_space = list(np.linspace(min_safety_budget, max_safety_budget, n_states)/30.0)
    n_epochs = 120    
    # ### checking the SimmerPID agent
    # from tf_algos.simmer_agents.pid_simmer_agent import SimmerPIDAgent, reference_generator, simmer_pid_params_cfg
    # simmer_pid_params_cfg['refs'] = reference_generator(
    #     ref_space=state_space, 
    #     n_epochs=n_epochs, 
    #     ref_type="simple"
    # )
    # simmer_pid_params_cfg['Kp'] = 2
    # simmer_pid_params_cfg['Ki'] = 1
    # simmer_pid_params_cfg['Kaw'] = 0.1
    # simmer_agent = SimmerPIDAgent(
    #     state_space=state_space,
    #     action_space=[-1, 1],
    #     trial_length=n_epochs,
    #     agent_params=simmer_pid_params_cfg
    # )
    # violations = np.random.randn(n_epochs)
    # actions = []
    # cur_violation = 0
    # for t_idx in range(n_epochs):        
    #     action = simmer_agent.act(cur_violation)
    #     actions.append(action)
    #     cur_violation = simmer_agent.current_ref + violations[t_idx]

    # ### checking the SimmerQ agent 
    from tf_algos.simmer_agents.q_simmer_agent import SimmerQAgent, simmer_q_params_cfg     
    simmer_q_params_cfg['state_space'] = state_space
    simmer_q_params_cfg['action_space'] = [-1, 0, 1]
    simmer_q_params_cfg['trial_length'] = n_epochs

    simmer_agent = SimmerQAgent(**simmer_q_params_cfg)
    violations = (np.random.rand(n_epochs)-0.8) * 0.2
    cur_violation = 0
    actions = []
    observed_states = []
    for t_idx in range(n_epochs):        
        action = simmer_agent.act(simmer_agent.state + cur_violation)
        actions.append(action)
        cur_violation = simmer_agent.state # + violations[t_idx]
        observed_states.append(cur_violation)
    print("done")

