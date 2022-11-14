import os
import torch
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.dynamics import EnsembleTransition

REWARD_GENERALIZATION_LIMIT = 1.1
STATE_GENERALIZATION_LIMIT = 11.


class EnsembleTrainer(object):
    def __init__(self, state_dim, action_dim, predict_reward, args):
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.predict_reward = predict_reward
        self.dynamics_path = args.dynamics_path
        self.dynamics_save_path = args.dynamics_save_path

        self.start_epoch = 0
        self.transition_num_epoch = args.transition_num_epoch
        self.batch_size = args.transition_batch_size

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args

        self.transition = EnsembleTransition(state_dim, action_dim, args.transition_layer_size, args.transition_layers,
                                             args.ensemble_size, predict_reward=predict_reward).to(self.device)
        self.transition_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.transition.parameters()),
                                                  lr=args.transition_lr,
                                                  weight_decay=0.000075)

        data_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter('./log/dynamics/{}_{}'.format(data_time, args.task))

    def train(self, train_buffer):
        if self.dynamics_path is not None and os.path.exists(self.dynamics_path + '/model.pt'):
            state = torch.load(self.dynamics_path + '/model.pt', map_location=self.device)
            ensemble_size = state['transition']['output_layer.bias'].shape[0]
            if self.transition.ensemble_size != ensemble_size:
                self.transition = EnsembleTransition(self.state_dim, self.action_dim, self.args.transition_layer_size,
                                                     self.args.transition_layers,
                                                     ensemble_size, predict_reward=self.predict_reward).to(self.device)
                self.transition_optim = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.transition.parameters()),
                    lr=self.args.transition_lr,
                    weight_decay=0.000075)
            self.transition.load_state_dict(state['transition'])

            epoch = state['epoch']
            with open(self.dynamics_path + '/reward_info.npy', 'rb') as f:
                self.transition.reward_shift = np.load(f)
                self.transition.reward_scale = np.load(f)
            if epoch < self.transition_num_epoch:
                self.start_epoch = epoch
                print('\nThe saved model is trained with {} epochs, the desired model is with {} epochs.\n'
                      'Continue training...'
                      .format(epoch, self.transition_num_epoch))
                self.transition_optim.load_state_dict(state['optimizer'])
                self.train_transition(train_buffer)
                state = {'transition': self.transition.state_dict(),
                         'optimizer': self.transition_optim.state_dict(),
                         'epoch': self.transition_num_epoch}
                torch.save(state, self.dynamics_path + '/model.pt')
            elif epoch > self.transition_num_epoch:
                print('\nThe saved model is trained with {} epochs, but the desired model is with {} epochs.'
                      .format(epoch, self.transition_num_epoch))
            self.transition.reset_ensemble_size(self.args.ensemble_size)
        elif self.dynamics_path is not None:
            print('\nModel training from scratch...')
            self.train_transition(train_buffer)
            if not os.path.exists(self.dynamics_path):
                os.makedirs(self.dynamics_path)
            state = {'transition': self.transition.state_dict(),
                     'optimizer': self.transition_optim.state_dict(),
                     'epoch': self.transition_num_epoch}
            torch.save(state, self.dynamics_path + '/model.pt')
            with open(self.dynamics_path + '/reward_info.npy', 'wb') as f:
                np.save(f, self.transition.reward_shift)
                np.save(f, self.transition.reward_scale)
        else:
            if self.dynamics_save_path is None:
                quit('\ndynamics_save_path is None.')
            print('\nModel training from scratch...')
            self.train_transition(train_buffer)
            if not os.path.exists(self.dynamics_save_path):
                os.makedirs(self.dynamics_save_path)
            state = {'transition': self.transition.state_dict(),
                     'optimizer': self.transition_optim.state_dict(),
                     'epoch': self.transition_num_epoch}
            torch.save(state, self.dynamics_save_path + '/model.pt')
            with open(self.dynamics_save_path + '/reward_info.npy', 'wb') as f:
                np.save(f, self.transition.reward_shift)
                np.save(f, self.transition.reward_scale)
        self.transition.requires_grad_(False)

    def train_transition(self, buffer, val_ratio=None):
        state = torch.tensor(buffer['obs'], device=self.device)
        action = torch.tensor(buffer['act'], device=self.device)
        next_state = torch.tensor(buffer['obs_next'], device=self.device)

        data_size = len(state)
        val_size = 0 if val_ratio is None else min(int(data_size * val_ratio) + 1)
        train_size = data_size - val_size

        input = torch.cat([state, action], dim=-1)
        input_std, input_mean = torch.std_mean(input, dim=0)
        if self.transition.mode == 'local':
            output_std = torch.std(next_state-state, dim=0)
        else:
            output_std = torch.std(next_state, dim=0)
        label_min, label_max = next_state.min(dim=0)[0], next_state.max(dim=0)[0]
        pred_min = (label_min + label_max) / 2 - (label_max - label_min) / 2 * STATE_GENERALIZATION_LIMIT
        pred_max = (label_min + label_max) / 2 + (label_max - label_min) / 2 * STATE_GENERALIZATION_LIMIT
        pred_min_2 = (label_min + label_max) / 2 - (label_max - label_min) * (STATE_GENERALIZATION_LIMIT - 0.5)
        pred_max_2 = (label_min + label_max) / 2 + (label_max - label_min) * (STATE_GENERALIZATION_LIMIT - 0.5)
        if self.predict_reward:
            reward = torch.tensor(buffer['rew'], device=self.device)
            output = torch.cat([next_state, reward], dim=-1)
            reward_min, reward_max = reward.min(), reward.max()
            reward_shift = reward_min - (reward_max - reward_min) * (REWARD_GENERALIZATION_LIMIT - 1.) / 2
            reward_scale = (reward_max - reward_min) * REWARD_GENERALIZATION_LIMIT
            output[:, -1] = (output[:, -1] - reward_shift) / reward_scale
            output_std = torch.cat([output_std, torch.std(output[:, -1:], dim=0)], dim=0)
            self.transition.reset_statistics(input_mean, input_std, output_std,
                                             pred_min, pred_max, pred_min_2, pred_max_2,
                                             reward_shift.cpu().numpy(), reward_scale.cpu().numpy()
                                             )
        else:
            self.transition.reset_statistics(input_mean, input_std, output_std,
                                             pred_min, pred_max, pred_min_2, pred_max_2
                                             )
            output = next_state

        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_input, train_output = input[train_splits.indices], output[train_splits.indices]
        val_input, val_output = input[val_splits.indices], output[val_splits.indices]
        num_batch = int(np.ceil(train_input.shape[0] / self.batch_size))

        for i_epoch in range(self.start_epoch, self.transition_num_epoch):
            idxs = np.random.randint(train_input.shape[0], size=[self.transition.ensemble_size, train_input.shape[0]])
            loss = 0.
            for i_batch in tqdm(range(num_batch), desc="{}th Epoch".format(i_epoch)):
                batch_idxs = idxs[:, i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
                loss += self.update_transition(train_input[batch_idxs], train_output[batch_idxs], beta=0.5)
            self.writer.add_scalar("Train/loss", loss, i_epoch)
            print("  Loss: {}".format(round(loss, 4)))

            if (val_ratio is not None) and (i_epoch % 5 == 0):
                loss = self.eval_transition(val_input, val_output)
                for k in range(len(loss)):
                    self.writer.add_scalar("Monitor/loss_" + str(k), loss[k], i_epoch)
                self.writer.add_scalar("Test/average_loss", np.mean(loss), i_epoch)
        print("Dynamics Model Trained.\n")

    def update_transition(self, input, output, beta=0.):
        dist = self.transition(input)
        loss = - dist.log_prob(output)

        weight = dist.scale.detach() ** (2 * beta)
        weight /= weight.mean(-2, keepdim=True)
        loss *= weight

        loss = loss.sum(0).mean()
        loss = loss + 0.01 * self.transition.max_logstd.sum() - 0.01 * self.transition.min_logstd.sum()

        self.transition_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transition.parameters(), 10.)
        self.transition_optim.step()
        return loss.detach().cpu().item()

    def eval_transition(self, input, output):
        with torch.no_grad():
            dist = self.transition(input)
            loss = - dist.log_prob(output).mean([-2, -1])
            return loss.cpu().numpy()
