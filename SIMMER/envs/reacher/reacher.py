from gym import spaces
from gym.utils import seeding
from typing import Tuple, Dict, List

from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np
from envs.wrappers.saute_env import saute_env
from envs.wrappers.safe_env import SafeEnv

reacher_cfg = {
    'action_dim': 1,
    'action_range': [-2, 2],
    'unsafe_reward': -3.75,
    'saute_discount_factor':1.0,
    'max_ep_len': 50,
    'min_rel_budget':1.0,
    'max_rel_budget':1.0,
    'test_rel_budget':1.0,
    'max_safety_budget':1.0, 
    'use_reward_shaping':True,
    'use_state_augmentation':True    
}


class CustomReacherEnv(ReacherEnv):
    """Custom reacher."""
    def __init__(
            self,
            mode: str = "train"
        ):
        self.observation_space_high = np.array(
            [np.pi, np.pi, np.pi, np.pi, 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)  # TODO: figure out
        self.observation_space = spaces.Box(low=-self.observation_space_high, high=self.observation_space_high)
        self.target_position = np.array([0, 0, 0])
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be deterministic, test or train"
        self._mode = mode
        self.seed()
        super(CustomReacherEnv, self).__init__()

    def seed(self, seed:int=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        vec = self.get_body_com("fingertip") - self.target_position
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl        
        self.do_simulation(action, self.frame_skip)
        next_state = self._get_obs() # next state
        done = False        
        return next_state, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset(self) -> np.ndarray:
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        self.goal = np.array([1.0, 1.0])
        self.target_position = np.concatenate([self.goal, [.01]])

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.target_position,
            ]
        )


class SafeReacherEnv(SafeEnv, CustomReacherEnv):
    safety_center = np.array([[.50, .50, 0.0]])   
    
    def _safety_cost_fn(self, state:np.ndarray, action:np.ndarray, next_state:np.ndarray) -> np.ndarray:
        """Computes the safety cost."""
        safety_vec = self.get_body_com("fingertip") - self.safety_center
        dist = np.linalg.norm(safety_vec)
        if dist<0.5:
            #Linearly increasse from 0 to 100 based on distance 
            return (1.0 - dist * 2.) * 100.0
        else:
            return 0

    def __str__(self):
        return "Safe Reacher with position constraints"


@saute_env
class SautedReacherEnv(SafeReacherEnv):
    """Sauted safe reacher."""

    def __str__(self):
        return "Sauted Reacher with position constraints"
