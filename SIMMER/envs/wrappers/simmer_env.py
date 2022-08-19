import numpy as np
import torch
from gym import Env
from gym import spaces
from envs.utils import Array

class SimmerBaseEnv(Env):
    def __init__(
            self,
            saute_discount_factor:float=0.99,
            max_ep_len:int=200,
            unsafe_reward:float=0,
            use_reward_shaping:bool=True, # ablation
            use_state_augmentation:bool=True, # ablation
    ):
            assert saute_discount_factor > 0 and saute_discount_factor <= 1, "Please specify a discount factor in (0, 1]" 
            assert max_ep_len > 0

            self.use_reward_shaping = use_reward_shaping
            self.use_state_augmentation = use_state_augmentation
            self.max_ep_len = max_ep_len

            self._saute_discount_factor = saute_discount_factor
            self._unsafe_reward = unsafe_reward 
            self._safety_state = None
            self.wrap = None
            self._max_safety_budget = 1.0
            self._rel_safety_budget = 1.0
            self._safety_budget = 1.0

    @property
    def max_safety_budget(self):
        return self._max_safety_budget 

    @max_safety_budget.setter
    def max_safety_budget(self, val):
        self._max_safety_budget = np.float32(val)
        if self.saute_discount_factor < 1:
             self._max_safety_budget *= (1 - self.saute_discount_factor ** self.max_ep_len) / (1 - self.saute_discount_factor) / self.max_ep_len
        self._rel_safety_budget = self._safety_budget / self._max_safety_budget

    @property
    def safety_budget(self):
        return self._safety_budget 

    @safety_budget.setter
    def safety_budget(self, val):
        self._safety_budget = np.float32(val)
        if self.saute_discount_factor < 1:
             self._safety_budget *= (1 - self.saute_discount_factor ** self.max_ep_len) / (1 - self.saute_discount_factor) / self.max_ep_len
        self._rel_safety_budget = self._safety_budget / self._max_safety_budget

    @property
    def saute_discount_factor(self):
        return self._saute_discount_factor 

    @property
    def unsafe_reward(self):
        return self._unsafe_reward

    def _augment_state(self, state:np.ndarray, safety_state:np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, safety_state]) if self.use_state_augmentation else state
        return augmented_state

    def safety_step(self, cost:np.ndarray) -> np.ndarray:
        """ Update the normalized safety state z' = (z - l / d) / gamma. """        
        self._safety_state -= cost / self.max_safety_budget
        self._safety_state /= self.saute_discount_factor
        return self._safety_state

    def reshape_reward(self, reward:Array, next_safety_state:Array) -> Array:
        raise NotImplementedError

    def step(self, action):
        """ Step through the environment. """
        next_obs, reward, done, info = self.wrap.step(action)        
        next_safety_state = self.safety_step(info['cost'])
        info['true_reward'] = reward         
        info['next_safety_state'] = next_safety_state
        reward = self.reshape_reward(reward, next_safety_state)
        augmented_state  =  self._augment_state(next_obs, next_safety_state)
        return augmented_state, reward, done, info

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        state = self.wrap.reset()
        if self.wrap._mode == "train":
            self._safety_state = self._rel_safety_budget
            # self._safety_state = self.wrap.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
        elif self.wrap._mode == "test" or self.wrap._mode == "deterministic":
            self._safety_state = self._rel_safety_budget
        else:
            raise NotImplementedError("this error should not exist!")
        augmented_state  =  self._augment_state(state,  self._safety_state)    
        return augmented_state  

    def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """ Compute rewards in a batch. """
        reward = self.wrap._reward_fn(states, actions, next_states, is_tensor=True) 
        if self.use_state_augmentation:
            # shape reward for model-based predictions 
            reward = self.reshape_reward(reward, next_states[:, -1].view(-1, 1))
        return reward

def simmer_env(cls):
    """ Class decorator for simmering an environment. """
    class SimmerEnv(SimmerBaseEnv):
        def __init__(
            self,
            max_safety_budget:float=1.0, 
            safety_budget:float=1.0, 
            saute_discount_factor:float=0.99,
            max_ep_len:int=200,
            unsafe_reward:float=0,
            use_reward_shaping:bool=True, # ablation
            use_state_augmentation:bool=True, # ablation
            **kwargs
        ):
            super().__init__(
                saute_discount_factor=saute_discount_factor, 
                max_ep_len=max_ep_len, 
                unsafe_reward=unsafe_reward,
                use_reward_shaping=use_reward_shaping,
                use_state_augmentation=use_state_augmentation
            )
            # wrapping the safe environment
            self.wrap = cls(**kwargs)

            # dealing with safety budget variables
            assert max_safety_budget > 0 and safety_budget > 0, "Please specify a positive safety budget" 
            self.safety_budget = np.float32(safety_budget)
            self.max_safety_budget = np.float32(max_safety_budget)

            # safety state definition 
            self._safety_state = self._rel_safety_budget

            # space definitions
            self.action_space = self.wrap.action_space
            self.obs_high = self.wrap.observation_space.high
            self.obs_low = self.wrap.observation_space.low
            if self.use_state_augmentation:
                self.obs_high = np.array(np.hstack([self.obs_high, np.inf]), dtype=np.float32)
                self.obs_low = np.array(np.hstack([self.obs_low, -np.inf]), dtype=np.float32)
            self.observation_space = spaces.Box(high=self.obs_high, low=self.obs_low)


        def reshape_reward(self, reward:Array, next_safety_state:Array) -> Array:
            """ Reshaping the reward. """
            if self.use_reward_shaping:
                reward = reward * (next_safety_state > 0) + self.unsafe_reward * (next_safety_state <= 0)
            return reward
    return SimmerEnv
