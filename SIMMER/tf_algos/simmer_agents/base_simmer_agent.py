from collections import deque
from typing import List
import numpy as np

class BaseSimmerAgent:
    """Base class for Simmering an algorithm, that is, deciding the current safety budget."""
    def __init__(
        self, 
        state_space:List, # finite state space for safety problems 
        action_space:List,
        trial_length:int, # trial length for the simmer agent
        **kwargs:dict
    ):
        """ 
        Initinilizing the simmer agent.
        """        
        assert state_space is not None, "Please specify the state space for the Simmer agent"
        assert trial_length > 0, "History length shoud be positive"
        assert type(action_space) == list, "entry action_space should be a list"
        self.trial_length = trial_length
        self.state_space = state_space
        self.action_space = action_space        
        # debug history
        self.error_history = deque([], maxlen=self.trial_length)         
        self.reward_history = deque([], maxlen=self.trial_length)     
        self.state_history = deque([], maxlen=self.trial_length)     
        self.action_history = deque([], maxlen=self.trial_length)
        self.observation_history = deque([], maxlen=self.trial_length)        

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def act(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reward_fn(self, observation:float, state:float, action:int):
        raise NotImplementedError


