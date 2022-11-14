import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax, pad
from torch.optim import Adam

from utils.utils import soft_update, hard_update
from model.policy import GaussianPolicy, DoubleQNetwork, RealNvpPolicy

REAL_RATIO = 0.5


def cal_alpha(target_kld, qf, log_ratio=None, min_val=1e-3):
    if log_ratio is None:
        return torch.clip(
            (
                (qf * (qf / min_val).softmax(dim=0)).sum(dim=0)
                - qf.mean(dim=0)
            ) / target_kld,
            min=min_val
        )
    else:
        return torch.clip(
            (
                (qf * (log_ratio + qf / min_val).softmax(dim=0)).sum(dim=0)
                - (log_ratio.softmax(dim=0) * qf).sum(dim=0)
            ) / target_kld,
            min=min_val
        )


class OfflineAgent(object):
    def __init__(self, state_dim, action_space, real_memory, args, value_clip):
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.target_kld = args.target_kld
        self.automatic_alpha_tuning = args.automatic_alpha_tuning
        self.real_batch_size = args.real_batch_size
        self.MC_size_action = args.MC_size_action
        self.explore_ratio = args.explore_ratio

        self.value_clip = value_clip
        self.tau_value = args.tau_value
        self.tau_policy = args.tau_policy

        self.policy_type = args.policy_type
        self.det_policy = args.det_policy

        self.device = torch.device("cpu" if args.cpu else "cuda")

        self.real_memory = real_memory

        self.critic = DoubleQNetwork(state_dim, action_space.shape[0], args.agent_layer_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = DoubleQNetwork(state_dim, action_space.shape[0], args.agent_layer_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            self.policy_ref = GaussianPolicy(state_dim, action_space.shape[0], args.agent_layer_size, action_space).to(
                self.device)
            self.policy = GaussianPolicy(state_dim, action_space.shape[0], args.agent_layer_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)
        elif self.policy_type == "Real_NVP":
            self.policy_ref = RealNvpPolicy(state_dim, action_space.shape[0], action_space).to(self.device)
            self.policy = RealNvpPolicy(state_dim, action_space.shape[0], action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)
        else:
            quit('Policy type is not supported.')
        hard_update(self.policy_ref, self.policy)

    def estimate_v(self, state, MC_size_action=10, with_RT_critic=False, clip_Q=True):
        state_torch = torch.as_tensor(state, device=self.device)
        with torch.no_grad():
            if self.policy_type in ["Gaussian", "Real_NVP"]:
                if self.det_policy:
                    action, _ = self.policy_ref.sample(state_torch, det=True)
                    qf_target = self.critic_target(state_torch, action)
                    qf_target = qf_target.min(-1)[0]
                    if clip_Q:
                        qf_target = torch.clip(qf_target, 0., 1 / (1 - self.gamma))
                    if with_RT_critic:
                        qf = self.critic(state_torch, action)
                        qf = qf.min(-1)[0]
                        if clip_Q:
                            qf = torch.clip(qf, 0., 1 / (1 - self.gamma))
                        return qf_target, qf
                    else:
                        return qf_target
                else:
                    IS_size = int(MC_size_action // 2)
                    action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=MC_size_action - IS_size)
                    action_proposal, log_prob_proposal = self.policy.sample(state_torch, num=IS_size)
                    action = torch.cat([action_proposal, action_ref], dim=0)

                    log_prob_IS = self.policy_ref.cal_prob(action_proposal, state_torch)
                    log_IS_ratio = torch.cat([log_prob_IS - log_prob_proposal, log_prob_ref - log_prob_ref], dim=0)

                    state_repeated = state_torch.expand([MC_size_action, *([-1] * state_torch.ndim)])
                    qf_target = self.critic_target(state_repeated, action)
                    qf_target = qf_target.min(-1)[0]
                    if clip_Q:
                        qf_target = torch.clip(qf_target, 0., 1 / (1 - self.gamma))

                    alpha = cal_alpha(self.target_kld, qf_target, log_IS_ratio) if self.automatic_alpha_tuning \
                        else self.alpha
                    qf_alpha_target = qf_target / alpha
                    qf_alpha_target += log_IS_ratio
                    v_target = alpha * (qf_alpha_target.logsumexp(dim=0) - log_IS_ratio.logsumexp(dim=0))

                    if with_RT_critic:
                        qf = self.critic(state_repeated, action)
                        qf = qf.min(-1)[0]
                        if clip_Q:
                            qf = torch.clip(qf, 0., 1 / (1 - self.gamma))

                        alpha = cal_alpha(self.target_kld, qf, log_IS_ratio) if self.automatic_alpha_tuning \
                            else self.alpha
                        qf_alpha = qf / alpha
                        qf_alpha += log_IS_ratio
                        v = alpha * (qf_alpha.logsumexp(dim=0) - log_IS_ratio.logsumexp(dim=0))
                        return v_target, v
                    else:
                        return v_target
            else:
                quit('Policy type is not supported.')

    def offline_update_parameters(self, adv_state, adv_action, adv_q):
        adv_state_torch = torch.as_tensor(adv_state, device=self.device)
        adv_action_torch = torch.as_tensor(adv_action, device=self.device)

        real_state, real_action, real_reward, real_next_state, real_done = self.real_memory.sample(self.real_batch_size)
        real_state_torch = torch.as_tensor(real_state, device=self.device)
        real_next_state_torch = torch.as_tensor(real_next_state, device=self.device)
        real_action_torch = torch.as_tensor(real_action, device=self.device)
        real_reward_torch = torch.as_tensor(real_reward, device=self.device).squeeze(1)
        real_done_torch = torch.as_tensor(real_done, dtype=torch.float32, device=self.device).squeeze(1)

        adv_q_torch = torch.as_tensor(adv_q, device=self.device)

        v = self.estimate_v(real_next_state_torch, MC_size_action=self.MC_size_action, clip_Q=self.value_clip)
        real_q_value = real_reward_torch + (1. - real_done_torch) * self.gamma * v

        state_torch = torch.cat([real_state_torch, adv_state_torch], dim=0)
        action_torch = torch.cat([real_action_torch, adv_action_torch], dim=0)
        q_value = torch.cat([real_q_value, adv_q_torch], dim=0)
        if self.value_clip:
            q_value = torch.clip(q_value, 0., 1 / (1 - self.gamma))

        qf = self.critic(state_torch, action_torch)
        q_value = q_value.repeat([2, 1]).t()
        qf_loss = F.mse_loss(qf[:self.real_batch_size],
                             q_value[:self.real_batch_size], reduction='none'
                             ).mean(0).sum(0) * REAL_RATIO \
                  + F.mse_loss(qf[self.real_batch_size:],
                               q_value[self.real_batch_size:], reduction='none'
                               ).mean(0).sum(0) * (1. - REAL_RATIO)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.policy_type in ["Gaussian", "Real_NVP"]:
            with torch.no_grad():
                IS_size = int(self.MC_size_action // 2)
                action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=self.MC_size_action - IS_size)
                action_1, log_prob_1 = \
                    self.policy_ref.tanh_normal_sample(real_action_torch, std=0.1, num=IS_size)
                action_2, log_prob_2 = self.policy.sample(adv_state_torch, num=IS_size)
                action_proposal = torch.cat([action_1, action_2], dim=1)
                action = torch.cat([action_proposal, action_ref], dim=0)

                log_prob_proposal = torch.cat([log_prob_1, log_prob_2], dim=1)
                log_prob_IS = self.policy_ref.cal_prob(action_proposal, state_torch)
                log_IS_ratio = torch.cat([log_prob_IS - log_prob_proposal, log_prob_ref - log_prob_ref], dim=0)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf_pi = self.critic(state_repeated, action)
                qf_pi = qf_pi.min(-1)[0]
                if self.value_clip:
                    qf_pi = torch.clip(qf_pi, 0., 1 / (1 - self.gamma))

                alpha = cal_alpha(self.target_kld, qf_pi, log_IS_ratio) if self.automatic_alpha_tuning \
                    else self.alpha
                log_weight = qf_pi / alpha + log_IS_ratio
                weight = softmax(log_weight, dim=0)
        else:
            quit('Policy type is not supported.')

        policy_loss = - torch.mul(self.policy.cal_prob(action, state_torch, beta=0.5), weight).sum(0)
        policy_loss = policy_loss[:self.real_batch_size].mean() * REAL_RATIO \
                      + policy_loss[self.real_batch_size:].mean() * (1. - REAL_RATIO)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau_value)
        soft_update(self.policy_ref, self.policy, self.tau_policy)

        return qf_loss.item(), policy_loss.item()

    def act(self, state, mode=0):
        state_torch = torch.as_tensor(state, device=self.device).reshape([-1, state.shape[-1]])
        with torch.no_grad():
            if mode == 0:
                # importance-sample according to policy_ref(a|s)*exp(Q(s,a)/α), with exploration following policy(a|s)
                IS_size = int(self.MC_size_action // 2)
                bool_index = np.random.uniform(size=state_torch.shape[0]) > self.explore_ratio

                candi_action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=self.MC_size_action - IS_size)
                candi_action_proposal, log_prob_proposal = self.policy.sample(state_torch, num=IS_size)
                candi_action = torch.cat([candi_action_proposal, candi_action_ref], dim=0)

                log_prob_IS = self.policy_ref.cal_prob(candi_action_proposal, state_torch)
                log_IS_ratio = torch.cat([log_prob_IS - log_prob_proposal, log_prob_ref - log_prob_ref], dim=0)

                state_exploit = state_torch[bool_index].expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf = self.critic(state_exploit, candi_action[:, bool_index])
                qf = qf.min(-1)[0]

                alpha = cal_alpha(self.target_kld, qf, log_IS_ratio) if self.automatic_alpha_tuning else self.alpha
                log_weight = qf / alpha
                log_weight += log_IS_ratio

                action = torch.empty([state_torch.shape[0], candi_action.shape[-1]], device=self.device)
                index = torch.multinomial(softmax(log_weight.t(), dim=1), 1)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action[bool_index] = candi_action[:, bool_index].gather(dim=0, index=index_gather).squeeze(dim=0)
                action[~bool_index] = candi_action[0, ~bool_index]
            elif mode == 1:
                if self.policy_type == "Gaussian":
                    # policy_ref mean
                    action, _ = self.policy_ref.sample(state_torch, det=True)
                else:
                    # sample according to policy_ref(a|s)
                    action, _ = self.policy_ref.sample(state_torch)
            elif mode == 2:
                if self.policy_type == "Gaussian":
                    # policy mean
                    action, _ = self.policy.sample(state_torch, det=True)
                else:
                    # sample according to policy(a|s)
                    action, _ = self.policy.sample(state_torch)
            elif mode == 3:
                # sample according to policy_ref(a|s)*exp(Q(s,a)/α)
                candi_action, log_prob = self.policy_ref.sample(state_torch, num=self.MC_size_action)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf = self.critic(state_repeated, candi_action)
                qf = qf.min(-1)[0]

                alpha = cal_alpha(self.target_kld, qf) if self.automatic_alpha_tuning else self.alpha

                index = torch.multinomial(softmax((qf / alpha).t(), dim=1), 1)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action = candi_action.gather(dim=0, index=index_gather).squeeze(dim=0)
            elif mode == 4:
                # draw N ssamples according to policy_ref(a|s) and find argmax_a Q(s,a)
                candi_action, log_prob = self.policy_ref.sample(state_torch, num=self.MC_size_action)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf = self.critic(state_repeated, candi_action)
                qf = qf.min(-1)[0]

                index = torch.argmax(qf, dim=0)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action = candi_action.gather(dim=0, index=index_gather).squeeze(dim=0)
        return action.detach().cpu().numpy()
