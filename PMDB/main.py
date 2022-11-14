import gym
import d4rl
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import args_setting
from module.ensemble_trainer import EnsembleTrainer
from module.adversarial_dynamics import AdversarialDynamics
from module.offline_agent import OfflineAgent
from module.replay_memory import ReplayMemory
from utils.utils import setup_seed, Evaluator
import utils.static

args = args_setting()

env = gym.make(args.task)
env.seed(args.seed)
setup_seed(args.seed)

dataset = d4rl.qlearning_dataset(env.unwrapped)
state = dataset['observations']
action = dataset['actions']
next_state = dataset['next_observations']
reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
done = np.expand_dims(np.squeeze(dataset['terminals']), 1)

state_dim = env.observation_space.shape[0]
action_space = env.action_space
reward_func = None
done_func = utils.static[args.task.split('-')[0]].termination_fn

predict_reward = reward_func is None
ensemble_trainer = EnsembleTrainer(state_dim, action_space.shape[0], predict_reward, args)
ensemble_trainer.train({'obs':state, 'act':action, 'obs_next':next_state, 'rew':reward})
transition = ensemble_trainer.transition

real_memory = ReplayMemory(state.shape[0])
normalized_reward = transition.get_normalized_reward(reward)
real_memory.push(state, action, normalized_reward, next_state, done)
offline_agent = OfflineAgent(state_dim, action_space, real_memory, args, value_clip=predict_reward)

all_state = np.concatenate([state, next_state[~done.astype(bool).reshape([-1])]], axis=0)
adv_dyna = AdversarialDynamics(all_state, offline_agent, transition, args, done_func, reward_func)
test_adv_dyna = AdversarialDynamics(all_state, offline_agent, transition, args, done_func, reward_func)

data_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter('./log/policy/{}_{}_{}_{}'.format(
    data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
evaluator = Evaluator(env, test_adv_dyna, offline_agent, [1, 2, 3, 4], data_time, args)

num_thounsands = 0
num_updates = 0
state = adv_dyna.reset()
while 1:
    if args.eval is True:
        evaluator.eval(num_updates)

    for i_rollout in tqdm(range(1000), desc="{}th Thousand Steps".format(num_thounsands)):
        action = offline_agent.act(state)
        _, adv_q, _, _ = adv_dyna.step(action)

        critic_loss, policy_loss = offline_agent.offline_update_parameters(state, action, adv_q)

        writer.add_scalar('loss/critic', critic_loss, num_updates)
        writer.add_scalar('loss/policy', policy_loss, num_updates)

        num_updates += 1
        state = adv_dyna.state.cpu().numpy()

    num_thounsands += 1
    if num_updates == args.agent_num_steps:
        break

env.close()
