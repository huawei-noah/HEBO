from envs.mountain_car.mountain_car import SafeMountainCarEnv, SautedMountainCarEnv, mcar_cfg
from gym.envs import register

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafeMountainCar-v0',
    entry_point='envs.mountain_car:SafeMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)

register(
    id='SautedMountainCar-v0',
    entry_point='envs.mountain_car:SautedMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)
