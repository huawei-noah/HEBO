# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy

import numpy as np
import torch
from torch import nn as nn, distributed as dist
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

from nap.policies.transformer import MixedTypeTransformerModel, TransformerModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.log_softmax(sorted_logits, dim=-1).exp(), dim=-1)
    #
    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0
    #
    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits, indices_to_remove


class NAP(nn.Module):

    def __init__(self, observation_space, action_space, deterministic, options, dataparallel=False,
                 policy_net_masks=None, mixed_type_options=None):
        super(NAP, self).__init__()
        self.N_features = None  # has to be set in init_structure()
        self.fine_tune = False
        self.use_mcts = False
        self.top_k = options.get('top_k', 5)
        self.deterministic = deterministic
        self.dataparallel = dataparallel
        self.device = 'cpu'
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if torch.cuda.device_count() > 0:
            self.device = rank % torch.cuda.device_count()
        else:
            self.device = 'cpu'

        self.policy_net_masks = None
        if policy_net_masks is not None:
            self.policy_net_masks = policy_net_masks.clone().to(self.device)

        # initialize the network structure
        self.init_structure(observation_space=observation_space, action_space=action_space, options=options,
                            mixed_type_options=mixed_type_options)

        # initialize weights
        self.apply(self.init_weights)

    def init_structure(self, observation_space, action_space, options, mixed_type_options=None):
        self.N_features = observation_space.shape[1]

        # policy network
        self.N_features_policy = self.N_features

        # value network
        if "use_value_network" in options and options["use_value_network"]:
            self.use_value_network = True
        else:
            self.use_value_network = False

        if mixed_type_options is not None and mixed_type_options.get("mixed_type", False):
            self.policy_net = MixedTypeTransformerModel(d_in=self.N_features_policy, d_out=1,
                                                        arch_spec=options["arch_spec"],
                                                        use_value_network=self.use_value_network,
                                                        cat_dims=mixed_type_options["cat_dims"],
                                                        cat_alphabet=mixed_type_options["cat_alphabet"],
                                                        cat_alphabet_map=mixed_type_options["cat_alphabet_map"],
                                                        num_dims=mixed_type_options["num_dims"],
                                                        )
        else:
            self.policy_net = TransformerModel(d_in=self.N_features_policy, d_out=1, arch_spec=options["arch_spec"],
                                               use_value_network=self.use_value_network)
        if self.dataparallel:
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
            self.policy_net.to(device)
            self.policy_net = DDP(self.policy_net, find_unused_parameters=False)
        else:
            self.policy_net.to(self.device)

    def forward(self, states, y_train, masks, state_ix, max_n_train):
        assert states.dim() == 3
        assert states.shape[-1] == self.N_features - 2

        # policy network
        if self.fine_tune:
            assert states.shape[0] == 1
            self.policy_net_local = copy.deepcopy(self.policy_net)
            self.ft_optim = torch.optim.Adam(list(self.policy_net_local.x_encoder.parameters() + self.policy_net_local.y_encoder.parameters() +
                                              self.policy_net_local.transformer_encoder.parameters() + self.policy_net_local.bucket_decoder.parameters()),
                                             lr=1e-5)
            print("PP", states.shape, flush=True)
            breakpoint()
        else:
            logits, fpred, values = self.policy_net.forward(states, y_train, masks, state_ix, max_n_train)
            logits.squeeze_(2)

        # value network
        if self.use_value_network:
            values.squeeze_(1)
        else:
            values = torch.zeros(states.shape[0]).to(logits.device)

        return logits, fpred, values

    def af(self, full_state):
        state = torch.from_numpy(full_state[0][None].astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(full_state[1][None].astype(np.float32)).to(self.device)
        masks = full_state[2]
        state_ix = torch.from_numpy(full_state[3][None].astype(np.float32)).to(self.device)
        max_T = full_state[5]
        with torch.no_grad():
            out = self.forward(state, y_train, masks, state_ix, max_T)
        af = out[0].to("cpu").numpy().squeeze()

        return af

    def act(self, state, y_train, masks_index, state_ix, max_T, mask_shift=None, batch_forward_pass=False, n_batch=10):
        # here, state is assumed to contain a single state, i.e. no batch dimension
        state = state.unsqueeze(0)  # add batch dimension
        y_train = y_train.unsqueeze(0)
        state_ix = state_ix.unsqueeze(0)

        if self.policy_net_masks is not None:
            masks = self.policy_net_masks[masks_index - mask_shift]
            masks = masks.unsqueeze(0)
            masks = masks.to(device=state.device)
        else:
            masks = masks_index

        if batch_forward_pass:
            # take the first n_init and T place holders, copy them n_batch times and split the rest into batches
            # we need each batch to have the same n_init + T first rows
            logits = []
            value = []
            Xc = state[:, :max_T]
            Yc = y_train[:, :max_T]
            Xt = state[:, max_T:]
            Yt = y_train[:, max_T:]
            N = Xt.shape[1]
            batch_size = N // n_batch
            for b in range(n_batch):
                assert isinstance(masks, int)
                start = b * batch_size
                end = (b + 1) * batch_size if b < n_batch-1 else N
                Xtbatch = Xt[:, start:end]
                Ytbatch = Yt[:, start:end]
                state_batch = torch.cat((Xc, Xtbatch), dim=1)
                y_train_batch = torch.cat((Yc, Ytbatch), dim=1)
                out_batch = self.forward(state_batch, y_train_batch, masks, state_ix, max_T)
                logits.append(out_batch[0])
                value.append(out_batch[2])
            logits = torch.cat(logits, dim=-1)
            value = torch.cat(value, dim=-1)
        else:
            out = self.forward(state, y_train, masks, state_ix, max_T)
            logits = out[0]
            # fpred = out[1]
            value = out[2]

        if self.deterministic:
            # most simple implementation:
            # action = torch.argmax(logits)
            # else
            distr = Categorical(logits=logits)
            diff = (state[0, max_T:][None] == state[0, :masks_index][:, None]) #TODO: double check mask_shift
            diff = diff.all(-1).any(0)
            action = torch.argmax(distr.probs * (1-diff.int()))
            logprobs = distr.log_prob(action).squeeze(0)
            proba_mask = diff
        else:
            diff = (state[0, max_T:][None] == state[0, :masks_index][:, None]) # TODO: double check mask_shift
            diff = diff.all(-1).any(0)
            logits[0, diff] = float("-inf")
            logits[0], proba_mask = top_k_top_p_filtering(logits[0], top_k=self.top_k)
            proba_mask = torch.logical_or(diff, proba_mask)
            if torch.isnan(logits).any():
                print('STATE\n', state)
                print('DIFF\n', diff)
                raise RuntimeError(f"Encountered NANs in logits {logits}")
            distr = Categorical(logits=logits)
            # to sample the action, the policy uses the current PROCESS-local random seed, don't re-seed in pi.act
            action = distr.sample()
            logprobs = distr.log_prob(action).squeeze(0)

        return action.squeeze(0), value.squeeze(0), logprobs, proba_mask

    def predict_vals_logps_ents(self, states_x, states_y, states_mask, states_ix, state_request, states_mask_shift,
                                actions, proba_mask):
        assert actions.dim() == 1
        assert states_x.shape[0] == actions.shape[0]

        masks = self.policy_net_masks[states_mask - states_mask_shift].to(device=states_x.device)
        out = self.forward(states_x, states_y, masks, states_ix, state_request)
        logits = out[0]
        fmodel = out[1]
        values = out[2]

        # assert((proba_mask.int().sum(-1) < proba_mask.shape[1]).all())

        logits[proba_mask] = float('-inf')
        distr = Categorical(logits=logits)
        logprobs = distr.log_prob(actions)
        entropies = distr.entropy()

        return values, logprobs, entropies, fmodel

    def predict_fmodel(self, states_x, states_y, states_mask, states_ix, state_request):
        out = self.forward(states_x, states_y, states_mask, states_ix, state_request)
        fmodel = out[1]

        return fmodel

    def set_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def reset(self):
        pass

    @staticmethod
    def num_flat_features(x):
        return np.prod(x.size()[1:])

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.fill_(0.0)
