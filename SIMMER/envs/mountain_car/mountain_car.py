import numpy as np
import math
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from envs.wrappers.saute_env import saute_env
from envs.wrappers.safe_env import SafeEnv 

mcar_cfg = dict(
        action_dim=1, 
        action_range=[-1, 1], 
        unsafe_reward=-1.,
        saute_discount_factor=0.99,
        max_ep_len=100,
        min_rel_budget=1.0,
        max_rel_budget=1.0,
        test_rel_budget=1.0,
        slope=1.0,
        use_reward_shaping=True,
        use_state_augmentation=True
)


class OurMountainCarEnv(Continuous_MountainCarEnv):
    def step(self, action):
        position = prev_position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        # reward -= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity], dtype=np.float32).squeeze()
        return self.state, reward, done, {} 


class SafeMountainCarEnv(SafeEnv, OurMountainCarEnv):
    """Safe Mountain Car Environment."""
    def __init__(self, mode:int="train", **kwargs):
        self._mode = mode
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be determinstic, test or train"
        super().__init__(**kwargs)

    def _get_obs(self):
        return self.state.squeeze()
        
    def reset(self):        
        if self._mode == "train":
            # making our lives easier with random starts 
            self.state = np.array([
                self.np_random.uniform(low=-0.6, high=0.4), 
                self.np_random.uniform(low=-self.max_speed, high=self.max_speed)
            ])
        elif self._mode == "test":
            self.state = np.array([
                self.np_random.uniform(low=-0.6, high=-0.4), 
                0
            ])
        return self.state.squeeze()

    def _safety_cost_fn(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> np.ndarray:        
        """Computes a fuel cost on the mountain car"""
        return np.linalg.norm(actions)

    def __str__(self):
        return "Safe Mountain Car with fuel constraints"


@saute_env
class SautedMountainCarEnv(SafeMountainCarEnv):
    """Sauted safe mountain car."""
    def __str__(self):
        return "Sauted Mountain car with fuel constraints"
