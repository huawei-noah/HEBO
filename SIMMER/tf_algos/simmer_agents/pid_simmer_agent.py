from matplotlib.pyplot import hist
from tf_algos.simmer_agents.base_simmer_agent import BaseSimmerAgent
from typing import List
import numpy as np
import copy
from collections import deque

def reference_generator(ref_space:List, n_epochs:int, ref_type:str):
    ref_schedule = [None] * n_epochs
    n_refs = len(ref_space); 
    interval_length = n_epochs // n_refs 
    if ref_type == "increase" or ref_type == "decrease":
        if ref_type == "increase":
            ref_space.sort()
        elif ref_type == "decrease":
            ref_space.sort(reverse=True)
        for t_idx in range(n_epochs):
             # determining the current interval, extra time steps are merged into initial interval
            cur_interval = max(0, n_refs - (n_epochs - t_idx) // interval_length - 1)
            # assert cur_interval <= n_refs, "the reference is not properly computed!"
            ref_schedule[t_idx] =  ref_space[cur_interval]   
    else:         
        raise NotImplementedError
    return ref_schedule

simmer_pid_params_cfg = dict(Kp=1, Ki=0, Kaw=0, polyak_update=0.005, trial_length=100, refs=None, state_space=[0], action_space=[-1, 1], global_action_bounds=[0, 0])

class SimmerPIDAgent(BaseSimmerAgent):
    """Simmering using a PI controller."""
    def __init__(
        self,        
        state_space:List,
        action_space:List,
        trial_length:int, # trial length for the simmer agent
        **kwargs:dict,    
    ):
        
        super().__init__(state_space=state_space, action_space=action_space, trial_length=trial_length)
        self.Kp  = kwargs.get('Kp', 0)
        self.Ki = kwargs.get('Ki', 0)
        self.Kaw = kwargs.get('Kaw', 0)
        self.refs = copy.copy(kwargs['refs'])
        self.polyak_update = kwargs.get('polyak_update', 0.995)
        self.global_action_bounds = kwargs.get('global_action_space', [0, 0])
        if self.global_action_bounds == [0, 0]:
            self.global_action_bounds = [0, max(self.refs)] 
        assert 0 < self.polyak_update <= 1, "Polyak update is in (0, 1] interval."
        assert self.Kp >=0 and self.Ki >=0 and self.Kaw >=0, "Gains of the PI controller should be non-negative."
        assert len(self.action_space) == 2 , "Expecting a list of length 2 defining the lower and upper bounds for the continuous action space."        
        if self.Kaw > 0:
            assert self.Ki > 0, "Anti-windup should be active only if there's integral control." 
        assert len(self.refs) >= self.trial_length, "References should be defined for the whole trial length."
        self.current_ref = None
        self.sum_history = 0
        self.prev_action = 0
        self.prev_error = 0
        self.prev_raw_action = 0
        self.I_history = deque([], maxlen=100)            

    # internal model 
    def reset(self):
        """Resetting the internal state of the agent."""
        self.current_ref = None
        self.sum_history = 0
        self.prev_action = 0
        self.prev_error = 0
        self.prev_raw_action = 0

    def step(self, action:int=None)->float:
        """
        Updating the internal state of the Simmer agent. 
        In this case the steps are deterministically defined using the schedule refs."""
        if self.refs:
            self.current_ref = self.refs.pop(0)
        assert self.current_ref is not None, "Current reference point (state) is not specified"
        return self.current_ref

    # computing the greedy action
    def get_greedy_action(self, observation:float, state:float) -> float:
        """
        Compute an action given the observation and the reference value
        """         
        current_ref = state
        # current error 
        cur_error = current_ref - observation 
        # low-pass filter
        cur_error =  self.polyak_update * cur_error + (1 - self.polyak_update) * self.prev_error
        # history of errors (for debugging)
        self.error_history.extend([cur_error]) 
        self.observation_history.extend([observation])
        # computing the sum of errors 
        self.I_history.append(cur_error)
        self.sum_history = sum(self.I_history) 
        # Proportional part
        P_part = self.Kp * cur_error
        # Integral part
        I_part = self.Ki * self.sum_history
        # anti-windup part
        AW_part = self.Kaw * (self.prev_action - self.prev_raw_action) #* self.sum_history        
        # raw acion
        cur_raw_action = P_part + I_part + AW_part
        # clipping action  
        cur_action = np.clip(cur_raw_action, self.action_space[0], self.action_space[1])
        next_safety_budget = np.clip(current_ref + cur_action, self.global_action_bounds[0], self.global_action_bounds[1])
        cur_action = next_safety_budget - current_ref # true action after global clipping 
        # storing actions 
        self.prev_action, self.prev_raw_action, self.prev_error = cur_action, cur_raw_action, cur_error
        return next_safety_budget

    def act(self, observation: float) -> float:
        """Compute action given the current observation."""
        # getting the current reference 
        current_ref = self.step()
        # getting the greedy action 
        next_safety_budget = self.get_greedy_action(observation=observation, state=current_ref)
        return next_safety_budget

