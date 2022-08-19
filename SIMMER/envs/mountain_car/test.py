import sys
sys.path.append(".")
import envs.mountain_car
import gym

if __name__ == "__main__":
    env = gym.make('SautedMountainCar-v0', safety_budget=50.0)
    env.reset()
    states, actions, next_states, rewards, dones, infos = [env.reset()], [], [], [], [], []
    for _ in range(100):
        a = env.action_space.sample()
        s, r, d, i = env.step(a)
        states.append(s)
        actions.append(a)
        next_states.append(s)
        rewards.append(r)
        dones.append(d)
        infos.append(i)
    print("dones")