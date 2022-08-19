import sys
sys.path.append(".")
import numpy as np
from envs.safety_gym.augmented_sg_envs import static_engine_cfg, dynamic_engine_cfg, BaselineEngine, AugmentedSafeEngine

if __name__ == '__main__':
    envs = [
        'StaticPointGoalEnv-v0', 'StaticCarGoalEnv-v0', 
        'DynamicPointGoalEnv-v0', 'DynamicCarGoalEnv-v0',
        'DynamicPointDoggoEnv-v0', 'DynamicDoggoGoalEnv-v0',
        'SautedStaticPointGoalEnv-v0', 'SautedDynamicPointGoalEnv-v0',
        'SautedStaticCarGoalEnv-v0', 'SautedDynamicCarGoalEnv-v0',
        'SautedStaticDoggoGoalEnv-v0', 'SautedDynamicDoggoGoalEnv-v0'
    ]
    def is_safety_gym_env(env_name):
        return ('Point' in  env_name or 'Car' in env_name or 'Doggo' in env_name) and \
                ('Goal' in  env_name) and \
                ('Static' in  env_name or 'Dynamic' in env_name)
    for env_name in envs:
        # env_name = envs[idx]
        # print(env_name)
        if is_safety_gym_env(env_name):
            if  'Static' in  env_name:
                engine_cfg = static_engine_cfg
            elif 'Dynamic' in env_name:       
                engine_cfg = dynamic_engine_cfg
            if 'Point' in  env_name:
                engine_cfg['robot_base'] = 'xmls/point.xml'
            elif 'Car' in env_name:       
                engine_cfg['robot_base'] = 'xmls/car.xml'
            elif 'Doggo' in env_name:       
                engine_cfg['robot_base'] = 'xmls/doggo.xml'
            if 'Goal' in  env_name:
                engine_cfg['task'] = 'goal'
            
        # if 'Sauted' in env_name:            
        #     env = AugmentedSafeEngine(saute_env_cfg, engine_cfg)        
        # else: 
        #     env = BaselineEngine(baseline_env_cfg, engine_cfg)
        import gym
        env = gym.make("SafePointGoal-v0", engine_cfg=engine_cfg)    
        print(env_name, env, env.config['robot_base'], env.config['task'])
        d = False
        min_reward = 0
        max_reward = 0
        rewards = []
        for _ in range(1):
            obs = env.reset()
            while not d:
                o, r, d, i = env.step(env.action_space.sample()) 
                rewards.append(r)                    
        print(np.mean(rewards), np.std(rewards), min(rewards), max(rewards))
