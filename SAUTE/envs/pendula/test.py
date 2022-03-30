import sys
sys.path.append(".")
import envs.pendula
import gym


if __name__ == "__main__":
   
    env = gym.make('SautedDoublePendulum-v0',safety_budget=1.,saute_discount_factor=0.1,mode="deterministic")
    env.reset()
    env2 = gym.make('SafeDoublePendulum-v0', mode="deterministic")
    env2.reset()  
    print(env.wrap._mode)
    states, actions, next_states, rewards, dones, infos = [env.reset()], [], [], [], [], []
    for _ in range(3000):
        a = env.action_space.sample()
        s, r, d, i = env.step(a)
        s2, r2, d2, i2 = env2.step(a)
        print(s, s2)
        print(r, r2)
        states.append(s)
        actions.append(a)
        next_states.append(s)
        rewards.append(r)
        dones.append(d)
        infos.append(i)
    print("dones")