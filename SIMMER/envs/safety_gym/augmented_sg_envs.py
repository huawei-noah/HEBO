import copy
import gym
import numpy as np
from typing import Tuple, Dict, List
from safety_gym.envs.engine import Engine
from gym.utils import seeding
from envs.wrappers.saute_env import saute_env

baseline_env_cfg = dict(
    action_dim=2,
    action_range=[-1, 1],
    max_ep_len=200,  # 200, # check?
    mode="train"
)

saute_env_cfg = dict(
    action_dim=2,
    action_range=[-1, 1],
    saute_discount_factor=0.99,
    safety_budget=15,
    unsafe_reward=-1.0,
    max_ep_len=200,  # 200, # check?
    min_rel_budget=1.,
    max_rel_budget=1.,
    test_rel_budget=1.,
    mode="train",
    use_reward_shaping=True,
    use_state_augmentation=True
)

static_engine_cfg = dict(
    placements_extents=[-1.5, -1.5, 1.5, 1.5],
    goal_size=0.3,
    goal_keepout=0.305,
    goal_locations=[(1.1, 1.1)],
    observe_goal_lidar=True,
    observe_hazards=True,
    constrain_hazards=True,
    lidar_max_dist=3,
    lidar_num_bins=16,
    hazards_num=1,
    hazards_size=0.7,
    hazards_keepout=0.705,
    hazards_locations=[(0, 0)]
)

dynamic_engine_cfg = dict(
    placements_extents=[-1.5, -1.5, 1.5, 1.5],
    goal_size=0.3,
    goal_keepout=0.305,
    observe_goal_lidar=True,
    observe_hazards=True,
    constrain_hazards=True,
    lidar_max_dist=3,
    lidar_num_bins=16,
    hazards_num=3,
    hazards_size=0.3,
    hazards_keepout=0.305
)


class BaselineEngine(Engine):
    """
    Base class for the safety gym environments 
    """

    def __init__(
            self,
            max_ep_len: int = 200,
            mode: str = "train",
            engine_cfg: Dict = None,
    ):
        super(BaselineEngine, self).__init__(engine_cfg)
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be deterministic, test or train"
        assert max_ep_len > 0
        self.max_episode_steps = max_ep_len
        self._mode = mode

    def seed(self, seed: int = None) -> List[int]:
        super(BaselineEngine, self).seed(seed)
        self.np_random, seed = seeding.np_random(self._seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, Dict]:
        obs, reward, done, info = super(BaselineEngine, self).step(action)
        info['pos_com'] = self.world.robot_com()  # saving position of the robot to plot
        return obs, reward, done, info

    def __str__(self):
        return "Safety Gym Engine with positional constraint"


@saute_env
class AugmentedSafeEngine(BaselineEngine):
    """Sauted engine using a wrapper"""

    def __str__(self):
        return "Sauted Safety Gym Engine with positional constraint"



