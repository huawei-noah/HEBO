from envs.pendula.single_pendulum import pendulum_cfg, SafePendulumEnv, SautedPendulumEnv, SimmeredPendulumEnv
from envs.pendula.double_pendulum import double_pendulum_cfg, SafeDoublePendulumEnv, SautedDoublePendulumEnv
from gym.envs import register

print('LOADING SAFE ENVIROMENTS') 

register(
    id='SafePendulum-v0',
    entry_point='envs.pendula:SafePendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SautedPendulum-v0',
    entry_point='envs.pendula:SautedPendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SimmeredPendulum-v0',
    entry_point='envs.pendula:SimmeredPendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SafeDoublePendulum-v0',
    entry_point='envs.pendula:SafeDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len']
)

register(
    id='SautedDoublePendulum-v0',
    entry_point='envs.pendula:SautedDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len']
)