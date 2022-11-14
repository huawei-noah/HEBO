import math
import torch
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from prettytable import PrettyTable


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def soft_clamp(x : torch.Tensor, _min=None, _max=None, _floor_std=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    if _floor_std is not None:
        x = torch.log(torch.exp(x) + _floor_std)
    return x


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Evaluator(object):
    def __init__(self, real_env, adv_env, agent, mode, data_time, args, steps=1000):
        self.real_env = real_env
        self.adv_env = adv_env
        self.agent = agent
        self.mode = mode
        self.steps = steps

        self.real_writer = SummaryWriter('./log/policy/{}_{}_{}_{}/real_eval'.format(
            data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
        self.adv_writer = SummaryWriter('./log/policy/{}_{}_{}_{}/adv_eval'.format(
            data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
        self.upper_writer = SummaryWriter('./log/policy/{}_{}_{}_{}/upper'.format(
            data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
        self.mean_writer = SummaryWriter('./log/policy/{}_{}_{}_{}/mean'.format(
            data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))
        self.lower_writer = SummaryWriter('./log/policy/{}_{}_{}_{}/lower'.format(
            data_time, args.task, args.policy_type, "autotune" if args.automatic_alpha_tuning else ""))

    def eval(self, num_updates):
        table = PrettyTable()
        table.add_column('Policy Type', ['Real Env', 'Adv Env'])

        for temp_mode in self.mode:
            reward_reg = []
            total_step = 0
            while total_step < self.steps:
                i_step = 0
                state = self.real_env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = self.agent.act(np.float32(state), mode=temp_mode)
                    state, reward, done, _ = self.real_env.step(action.ravel())
                    episode_reward += reward
                    i_step += 1
                total_step += i_step
                reward_reg.append(episode_reward)

            state = np.concatenate([self.real_env.reset().reshape([1, -1]) for _ in range(10)], 0)
            self.adv_env.reset(np.float32(state))
            index = np.arange(10)
            episode_reward = np.zeros(10)
            adv_q_target_reg = []
            adv_q_reg = []
            ensemble_std_reg = []
            for _ in range(self.steps):
                action = self.agent.act(self.adv_env.state, mode=temp_mode)
                feedback, adv_q_target, adv_q, ensemble_std = self.adv_env.step(action, test_mode=True)
                reward, _, done = feedback
                episode_reward[index] += self.adv_env.transition.get_raw_reward(reward)
                index = index[~done]
                adv_q_target_reg.append(adv_q_target.reshape(-1))
                adv_q_reg.append(adv_q.reshape(-1))
                ensemble_std_reg.append(ensemble_std)
                if index.shape[0] == 0:
                    break

            real_reward = np.mean(reward_reg)
            adv_reward = episode_reward.mean()

            if temp_mode == 1:
                self.real_writer.add_scalar('test/reward_1 (Reported in paper)', real_reward, num_updates)
                self.adv_writer.add_scalar('test/reward_1 (Reported in paper)', adv_reward, num_updates)
                table.add_column('1 (Reported)', [round(real_reward, 2), round(adv_reward, 2)])

                adv_q_target_reg = np.concatenate(adv_q_target_reg, 0)
                adv_q_reg = np.concatenate(adv_q_reg, 0)
                ensemble_std_reg = np.concatenate(ensemble_std_reg, 0)
                std_adv_q_target, mean_adv_q_target = np.std(adv_q_target_reg), np.mean(adv_q_target_reg)
                std_adv_q, mean_adv_q = np.std(adv_q_reg), np.mean(adv_q_reg)
                ensemble_logstd_reg = np.log(ensemble_std_reg)
                std_ensemble_logstd, mean_ensemble_logstd = np.std(ensemble_logstd_reg), np.mean(ensemble_logstd_reg)

                mean_adv_q_target = mean_adv_q_target
                std_adv_q_target = std_adv_q_target

                mean_adv_q = mean_adv_q
                std_adv_q = std_adv_q

                mean_ensemble_logstd = mean_ensemble_logstd
                std_ensemble_logstd = std_ensemble_logstd

                self.upper_writer.add_scalar('monitor/adv_q_target', mean_adv_q_target + std_adv_q_target, num_updates)
                self.mean_writer.add_scalar('monitor/adv_q_target', mean_adv_q_target, num_updates)
                self.lower_writer.add_scalar('monitor/adv_q_target', mean_adv_q_target - std_adv_q_target, num_updates)

                self.upper_writer.add_scalar('monitor/adv_q', mean_adv_q + std_adv_q, num_updates)
                self.mean_writer.add_scalar('monitor/adv_q', mean_adv_q, num_updates)
                self.lower_writer.add_scalar('monitor/adv_q', mean_adv_q - std_adv_q, num_updates)

                self.upper_writer.add_scalar('monitor/ensemble_std', mean_ensemble_logstd + std_ensemble_logstd,
                                             num_updates)
                self.mean_writer.add_scalar('monitor/ensemble_std', mean_ensemble_logstd, num_updates)
                self.lower_writer.add_scalar('monitor/ensemble_std', mean_ensemble_logstd - std_ensemble_logstd,
                                             num_updates)
            else:
                self.real_writer.add_scalar('test/reward_' + str(temp_mode), real_reward, num_updates)
                self.adv_writer.add_scalar('test/reward_' + str(temp_mode), adv_reward, num_updates)
                table.add_column(str(temp_mode), [round(real_reward, 2), round(adv_reward, 2)])
        print(table)
