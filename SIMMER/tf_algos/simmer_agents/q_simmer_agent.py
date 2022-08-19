from tf_algos.simmer_agents.base_simmer_agent import BaseSimmerAgent
from typing import List
import numpy as np
from collections import deque
import copy

simmer_q_params_cfg = dict(state_space=None, action_space=[-1, 0, 1], trial_length=100, epsilon_greedy=0.8, action_penalty=0, lr=0.03, simmer_discount_factor=0.9, reward_thresh=0.01)

class SimmerQAgent(BaseSimmerAgent):
    """Simmering using Q learning."""
    def __init__(
        self,
        state_space:List,
        action_space:List, # must contain a zero action
        trial_length:int,
        **kwargs:dict
    ):
        super().__init__(state_space=state_space, action_space=action_space, trial_length=trial_length)
        assert 0 in action_space, "We need a zero action in the action space."
        self.lr = kwargs.get("lr", 1e-2) 
        self.simmer_discount_factor = kwargs.get("simmer_discount_factor", 0.99)
        self.epsilon_greedy = kwargs.get("epsilon_greedy", 0.95)
        self.action_penalty = kwargs.get("action_penalty", 0.01)
        self.reward_thresh = kwargs.get("reward_thresh", 0.1)
        self.polyak_update = kwargs.get("polyak_update", 0.001)
        assert 0 < self.polyak_update <= 1, "Polyak update is in (0, 1] interval."

        self.n_states = len(self.state_space)
        self.n_actions = len(self.action_space)
        self.q_function = np.zeros((self.n_states, self.n_actions))
        self.initial_state_idx = 0
        self.state = self.state_space[self.initial_state_idx]
        self.action = 0
        self.step(self.action) # making a step in the environment
        self.prev_observation = copy.copy(self.state)
        ### observation filtering
        self.filtered_observation_history = []
        self.filtered_observation = 0

    # internal model 
    def reset(self):
        """Resetting the internal state of the agent."""
        self.state = self.state_space[self.initial_state_idx]
        self.step(self.action)
        self.action = 0 # self.get_random_action()

    def step(self, action:int)->float:
        """
        Updating the internal state of the Simmer agent. 
        The agent changes the index of the current state by the value of the current action 
        within the state-space, i.e., the values for the index are clipped between 0 and n_states-1
        """
        state_idx = self.get_state_idx(self.state)    
        state_idx = np.clip(state_idx + action, 0, self.n_states-1) 
        self.state = self.state_space[state_idx]
        return self.state  

    def reward_fn(self, state:float, action:int,  observation:float):
        """Reward for the interal agent's process."""        
        action_idx = self.action_space.index(action)
        if int(self.reward_thresh > observation - state > -self.reward_thresh) :
             reward = np.array([-1, 1, 0.5])[action_idx]
        elif int(observation - state <= -self.reward_thresh):
             reward = np.array([-1, 0.5, 2])[action_idx]
        elif int(observation - state >= self.reward_thresh):            
            reward = np.array([2, -1, -1])[action_idx]
        return reward 

    # agent's methods
    def get_action_idx(self, action:int) -> int:
        """
        Action value to action index
        """
        action_idx = self.action_space.index(action)
        return action_idx

    def get_state_idx(self, state:float) -> int:
        """
        State value to state index.
        """
        state_idx = self.state_space.index(state)
        return state_idx

    def get_random_action(self):
        """
        Generate a random action. 
        """
        action_idx = np.random.randint(low=0, high=len(self.action_space), size=1)[0]
        self.action = self.action_space[action_idx]
        return self.action
   
    def update_q(self, state:int, action:int, reward:int, next_state:int):
        """Updating the Q function."""
        action_idx = self.get_action_idx(action)
        state_idx = self.get_state_idx(state)
        next_state_idx  = self.get_state_idx(next_state)
        self.q_function[state_idx, action_idx] = (1 - self.lr) * self.q_function[state_idx, action_idx] + \
            self.lr * (reward + self.simmer_discount_factor * np.max(self.q_function[next_state_idx, :]))
        
    def get_greedy_action(self, observation:float, state:float) -> float:
        """Compute the greedy action in the given state."""
        state_idx = self.state_space.index(state)
        action_idx = np.argmax(self.q_function[state_idx,:]) 
        self.action = self.action_space[action_idx]       
        return self.action

    # applying the action
    def act(self, observation: float) -> float:   
        """
        Compute an action given an observation
        """ 
        prev_observation = self.filtered_observation
        # filtering an observation to avoid jitter
        self.filtered_observation = self.polyak_update * observation + (1 - self.polyak_update) * prev_observation 
        self.filtered_observation_history.append(self.filtered_observation)
        # tuple s_t, a_t, s_{t+1}, r_t, 
        state = self.state
        # getting a new action (eps-greedy)
        eps_ = np.random.random()
        if eps_ < self.epsilon_greedy:
            action = self.get_greedy_action(observation=self.filtered_observation, state=state)
        else:   
            action = self.get_random_action()        
        reward = self.reward_fn(state=state, action=action, observation=self.filtered_observation) 
        # saving for debug
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)
        # updating the interal state
        next_safety_budget = self.step(action)
        next_state = self.state
        # updating the policy
        self.update_q(state=state, action=action, reward=reward, next_state=next_state)        
        # getting new observations -->         
        return next_safety_budget
