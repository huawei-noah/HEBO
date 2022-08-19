import numpy as np
from gym.envs.mujoco import InvertedDoublePendulumEnv
from envs.wrappers.saute_env import saute_env
from envs.wrappers.safe_env import SafeEnv 

from typing import Dict, Tuple

double_pendulum_cfg = dict(
        action_dim=1,
        action_range=[
            -1, 
            1],
        unsafe_reward=-200.,
        saute_discount_factor=1.0,
        max_ep_len=200,
        min_rel_budget=1.0,
        max_rel_budget=1.0,
        test_rel_budget=1.0,
        use_reward_shaping=True,
        use_state_augmentation=True

)

class DoublePendulumEnv(InvertedDoublePendulumEnv):
    """Custom double pendulum."""
    def __init__(self, mode="train"):
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be deterministic, test or train"
        self._mode = mode
        super().__init__()

    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state, reward, done, info = super().step(action)
        reward /= 10. # adjusting the reward to match the cost
        return next_state, reward, done, info

 
class SafeDoublePendulumEnv(SafeEnv, DoublePendulumEnv):
    """Safe double pendulum."""    
    def __init__(self, **kwargs):
        self.unsafe_min = np.pi * (-25. / 180.)
        self.unsafe_max = np.pi * (75. / 180.)
        self.unsafe_middle = 0.5 * (self.unsafe_max + self.unsafe_min) 
        self.max_distance =  0.5 * (self.unsafe_max - self.unsafe_min) 
        super().__init__(**kwargs)

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:  
        """Computes a linear safety cost between the current position 
        (if its near the unsafe area, aka in the hazard region)
        and the centre of the unsafe region."""
        assert type(state) is np.ndarray and type(next_state) is np.ndarray and type(action) is np.ndarray, "Arguments must be np.ndarray"
        thetas = np.arctan2(state[..., 1], state[..., 3]) 
        dist_to_center = np.abs(self.unsafe_middle - thetas)
        unsafe_mask = np.float64(((self.unsafe_min) <= thetas) & (thetas <= (self.unsafe_max)))
        costs = ((self.max_distance - dist_to_center) / (self.max_distance)) * unsafe_mask
        return costs

    def __str__(self):
        return "Safe Double Pendulum with angle constraints"

@saute_env
class SautedDoublePendulumEnv(SafeDoublePendulumEnv):
    """Sauted safe double pendulum."""
    def __str__(self):
        return "Sauted Double Pendulum with angle constraints"

