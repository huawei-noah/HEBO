from envs.safety_gym.augmented_sg_envs import BaselineEngine, baseline_env_cfg, static_engine_cfg
from gym.envs import register

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafePointGoal-v0',
    entry_point='envs.safety_gym.augmented_sg_envs:BaselineEngine',
    max_episode_steps=baseline_env_cfg['max_ep_len'],
)

