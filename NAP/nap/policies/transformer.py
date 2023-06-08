# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from nap.RL.botorch_helper import logEI, EI
from nap.policies.bar_distribution import BarDistribution, get_bucket_limits


def generate_D_q_matrix(sz, train_size, device='cpu'):
    mask = torch.zeros((sz, sz), device=device) == 0
    mask[:, train_size:].zero_()
    mask |= torch.eye(sz, device=device) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):
    def __init__(self, d_in, d_out, arch_spec=None, use_value_network=False):
        super().__init__()

        # if arch_spec is None:
        #     arch_spec = dict(nbuckets=100, emb_size=512, nlayers=6, nhead=4, dropout=0.0)

        self.use_value_network = use_value_network
        self.nhead = arch_spec['nhead']
        self.d_in = d_in
        self.nbuckets = arch_spec['nbuckets']
        self.temperature = arch_spec['temperature']
        self.af_name = arch_spec['af_name']
        assert self.af_name in ['ei', 'ucb', 'mlp', 'none']
        self.joint_model_af_training = arch_spec.get('joint_model_af_training', True)

        if 'y_range' not in arch_spec.keys():
            arch_spec['y_range'] = 8.

        if self.nbuckets > 2:
            if isinstance(arch_spec['y_range'], float):
                self.buckets = BarDistribution(borders=get_bucket_limits(self.nbuckets,
                                                                full_range=(-arch_spec['y_range'], arch_spec['y_range'])))
            elif isinstance(arch_spec['y_range'], tuple):
                self.buckets = BarDistribution(borders=get_bucket_limits(self.nbuckets,
                                                                         full_range=(
                                                                         arch_spec['y_range'][0], arch_spec['y_range'][1])))
            else:
                raise ValueError(f"Unknow y_range type: {arch_spec['y_range']}")
        if 'dim_feedforward' not in arch_spec.keys():
            arch_spec['dim_feedforward'] = arch_spec['emb_size']

        encoder_layers = TransformerEncoderLayer(arch_spec['emb_size'], arch_spec['nhead'], arch_spec['dim_feedforward'],
                                                 arch_spec['dropout'], activation='gelu', batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, arch_spec['nlayers'])

        self.x_encoder = nn.Linear(self.d_in - 2, arch_spec['emb_size'])
        self.y_encoder = nn.Linear(1, arch_spec['emb_size'])
        self.bucket_decoder = nn.Sequential(nn.Linear(arch_spec['emb_size'], arch_spec['dim_feedforward']),
                                            nn.GELU(), nn.Linear(arch_spec['dim_feedforward'], arch_spec['nbuckets']))
        if self.af_name == "mlp":
            self.af_decoder = nn.Sequential(nn.Linear(arch_spec['nbuckets'] + arch_spec['emb_size'] + 2, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, 200), nn.ReLU(),
                                            nn.Linear(200, 200), nn.ReLU(),
                                            nn.Linear(200, 1))

        if self.use_value_network:
            self.value_head = nn.Sequential(nn.Linear(2, arch_spec['emb_size']),
                                        nn.Tanh(), nn.Linear(arch_spec['emb_size'], 1))

    def forward(self, x_src, y_src, src_mask, state_ix, single_eval_pos):
        if torch.is_tensor(single_eval_pos):
            assert single_eval_pos[0] == single_eval_pos[-1]
            single_eval_pos = single_eval_pos[0]
        if not torch.is_tensor(src_mask):
            src_mask = generate_D_q_matrix(x_src.shape[1], src_mask, device=x_src.device).unsqueeze(0).repeat(x_src.shape[0], 1, 1)

        # x_src = torch.cat((x_src, state_ix[:, None].repeat(1, x_src.shape[1], 1)), -1)
        x_emb = self.x_encoder(x_src)
        y_emb = self.y_encoder(y_src.to(x_src))
        train_x = x_emb[:, :single_eval_pos] + y_emb[:, :single_eval_pos]

        src = torch.cat([train_x, x_emb[:, single_eval_pos:]], 1)

        output = self.transformer_encoder(src, src_mask.repeat_interleave(self.nhead, 0))

        output_f = self.bucket_decoder(output[:, single_eval_pos:])

        if self.af_name == "ei":
            assert state_ix.shape[1] == 2  # assume first state dim is best y so far
            if self.nbuckets == 1:
                output_af = torch.maximum(output_f - state_ix[:, 0, None, None], torch.zeros_like(output_f))
            elif self.nbuckets == 2:
                mean = output_f[:, :, 0]
                sigma = torch.nn.functional.softplus(output_f[:, :, 1])
                output_af = EI(mean, sigma, state_ix[:, 0, None]).unsqueeze(-1)
            else:
                output_af = self.buckets.ei_batch(output_f, state_ix[:, 0, None, None]).unsqueeze(-1)
        elif self.af_name == "ucb":
            if self.nbuckets == 1:
                output_af = output_f
            elif self.nbuckets == 2:
                mean = output_f[:, :, 0]
                sigma = torch.nn.functional.softplus(output_f[:, :, 1])

                output_af = (mean + 2.0 * sigma).unsqueeze(-1)
            else:
                output_af = self.buckets.ucb(output_f, 2.0, True).unsqueeze(-1)
        elif self.af_name == "mlp":
            state_ix_af = state_ix.unsqueeze(1).repeat(1, output_f.shape[1], 1)
            if self.nbuckets == 1:
                raise()
            elif self.nbuckets == 2:
                if self.joint_model_af_training:
                    mean = output_f[:, :, 0:1]
                    sigma = torch.nn.functional.softplus(output_f[:, :, 1:2])
                else:
                    mean = output_f[:, :, 0:1].detach()
                    sigma = torch.nn.functional.softplus(output_f[:, :, 1:2].detach())

                output_af = self.af_decoder(torch.cat((mean, sigma, state_ix_af), -1))
            else:
                # if joint training, keep gradients from acqf to model, otherwise detach
                if self.joint_model_af_training:
                    buckets = output_f.softmax(-1)
                    x_in = x_emb[:, single_eval_pos:]
                else:
                    buckets = output_f.detach().softmax(-1)
                    x_in = x_emb[:, single_eval_pos:].detach()
                output_af = self.af_decoder(torch.cat((buckets, x_in, state_ix_af), -1))
        elif self.af_name == "none":
            output_af = output_f


        # entropy correction with temperature
        output_af = output_af / self.temperature
        # output_af = torch.ones_like(output_af)

        if self.use_value_network:
            output_vf = self.value_head(state_ix)
            return output_af, output_f, output_vf

        return output_af, output_f, None


class MixedTypeTransformerModel(nn.Module):
    def __init__(self, d_in, d_out, arch_spec=None, use_value_network=False,
                 num_dims=None, cat_dims=None, cat_alphabet=None, cat_alphabet_map=None):
        super().__init__()

        self.use_value_network = use_value_network
        self.nhead = arch_spec['nhead']
        self.d_in = d_in
        self.nbuckets = arch_spec['nbuckets']
        self.temperature = arch_spec['temperature']
        self.af_name = arch_spec['af_name']
        assert self.af_name in ['ei', 'ucb', 'mlp', 'none']
        self.joint_model_af_training = arch_spec.get('joint_model_af_training', True)

        self.cat_dims = cat_dims
        self.num_dims = num_dims
        self.cat_alphabet = cat_alphabet
        self.cat_alphabet_map = cat_alphabet_map
        self.d_cat = len(cat_dims) if cat_dims is not None else 0

        if cat_dims is None:
            raise RuntimeError(f"cat_dims is None.\n"
                               f"If using only numerical dimensions, use the regular TransformerModel instead.\n"
                               f"If using categorical dims, please provide them as well as their alphabet.")
        if num_dims is None:
            dims = np.arange(d_in)
            self.num_dims = [i for i in dims if i not in self.cat_dims]
        self.d_num = len(self.num_dims)

        if 'y_range' not in arch_spec.keys():
            arch_spec['y_range'] = 8.

        if self.nbuckets > 2:
            if isinstance(arch_spec['y_range'], float):
                self.buckets = BarDistribution(borders=get_bucket_limits(self.nbuckets,
                                                                         full_range=(
                                                                         -arch_spec['y_range'], arch_spec['y_range'])))
            elif isinstance(arch_spec['y_range'], tuple):
                self.buckets = BarDistribution(borders=get_bucket_limits(self.nbuckets,
                                                                         full_range=(
                                                                             arch_spec['y_range'][0],
                                                                             arch_spec['y_range'][1])))
            else:
                raise ValueError(f"Unknow y_range type: {arch_spec['y_range']}")

        if 'dim_feedforward' not in arch_spec.keys():
            arch_spec['dim_feedforward'] = arch_spec['emb_size']

        encoder_layers = TransformerEncoderLayer(arch_spec['emb_size'], arch_spec['nhead'], arch_spec['dim_feedforward'],
                                                 arch_spec['dropout'], activation='gelu', batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, arch_spec['nlayers'])

        if self.d_num > 0:
            self.xnum_encoder = nn.Linear(self.d_num, arch_spec['emb_size'])
        else:
            self.xnum_encoder = None
        self.xcat_encoder = [nn.Linear(len(self.cat_alphabet[c]), arch_spec['emb_size']) for c in self.cat_dims]
        self.xcat_encoder = nn.ModuleList(self.xcat_encoder)
        self.y_encoder = nn.Linear(1, arch_spec['emb_size'])
        self.bucket_decoder = nn.Sequential(nn.Linear(arch_spec['emb_size'], arch_spec['dim_feedforward']),
                                            nn.GELU(), nn.Linear(arch_spec['dim_feedforward'], arch_spec['nbuckets']))

        if self.af_name == "mlp":
            self.af_decoder = nn.Sequential(nn.Linear(arch_spec['nbuckets'] + arch_spec['emb_size'] + 2, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, 200), nn.ReLU(),
                                            nn.Linear(200, 200), nn.ReLU(),
                                            nn.Linear(200, 1))

        if self.use_value_network:
            self.value_head = nn.Sequential(nn.Linear(2, arch_spec['emb_size']),
                                        nn.Tanh(), nn.Linear(arch_spec['emb_size'], 1))

    def forward(self, x_src, y_src, src_mask, state_ix, single_eval_pos):
        if torch.is_tensor(single_eval_pos):
            assert single_eval_pos[0] == single_eval_pos[-1]
            single_eval_pos = single_eval_pos[0]
        if not torch.is_tensor(src_mask):
            src_mask = generate_D_q_matrix(x_src.shape[1], src_mask, device=x_src.device).unsqueeze(0).repeat(x_src.shape[0], 1, 1)

        # x_src = torch.cat((x_src, state_ix[:, None].repeat(1, x_src.shape[1], 1)), -1)
        xnum_src, xcat_src  = x_src[..., self.num_dims], x_src[..., self.cat_dims]
        assert xnum_src.shape[-1] == self.d_num
        assert xcat_src.shape[-1] == self.d_cat

        xcat_emb = []
        for d, c in enumerate(self.cat_dims):
            xcat_src_d = xcat_src[..., d]
            if self.cat_alphabet[c][-1] > len(self.cat_alphabet[c]):
                # breakpoint()
                mapped = []
                for el in xcat_src_d.flatten():
                    if el != 0:
                        mapped.append(self.cat_alphabet_map[c][int(el)])
                    else:
                        mapped.append(0)
                xcat_src_d = torch.tensor(mapped).to(xcat_src_d).view(*xcat_src_d.shape)
                # breakpoint()
            xcat_1hot_d = torch.nn.functional.one_hot(xcat_src_d.to(int), num_classes=len(self.cat_alphabet[c])).to(x_src)
            xcat_enc_d = self.xcat_encoder[d](xcat_1hot_d)
            xcat_emb.append(xcat_enc_d)

        xcat_emb = torch.stack(xcat_emb, -2).sum(-2)
        if self.d_num > 0 and self.xnum_encoder is not None:
            xnum_emb = self.xnum_encoder(xnum_src)
            x_emb = xnum_emb + xcat_emb
        else:
            x_emb = xcat_emb
        y_emb = self.y_encoder(y_src.to(x_src))
        train_x = x_emb[:, :single_eval_pos] + y_emb[:, :single_eval_pos]

        src = torch.cat([train_x, x_emb[:, single_eval_pos:]], 1)

        output = self.transformer_encoder(src, src_mask.repeat_interleave(self.nhead, 0))

        output_f = self.bucket_decoder(output[:, single_eval_pos:])

        if self.af_name == "ei":
            assert state_ix.shape[1] == 2  # assume first state dim is best y so far
            if self.nbuckets == 1:
                output_af = torch.maximum(output_f - state_ix[:, 0, None, None], torch.zeros_like(output_f))
            elif self.nbuckets == 2:
                mean = output_f[:, :, 0]
                sigma = torch.nn.functional.softplus(output_f[:, :, 1])

                output_af = EI(mean, sigma, state_ix[:, 0, None]).unsqueeze(-1)
            else:
                output_af = self.buckets.ei_batch(output_f, state_ix[:, 0, None, None]).unsqueeze(-1)
        elif self.af_name == "ucb":
            if self.nbuckets == 1:
                output_af = output_f
            elif self.nbuckets == 2:
                mean = output_f[:, :, 0]
                sigma = torch.nn.functional.softplus(output_f[:, :, 1])

                output_af = (mean + 2.0 * sigma).unsqueeze(-1)
            else:
                output_af = self.buckets.ucb(output_f, 2.0, True).unsqueeze(-1)
        elif self.af_name == "mlp":
            state_ix_af = state_ix.unsqueeze(1).repeat(1, output_f.shape[1], 1)
            if self.nbuckets == 1:
                raise()
            elif self.nbuckets == 2:
                if self.joint_model_af_training:
                    mean = output_f[:, :, 0:1]
                    sigma = torch.nn.functional.softplus(output_f[:, :, 1:2])
                else:
                    mean = output_f[:, :, 0:1].detach()
                    sigma = torch.nn.functional.softplus(output_f[:, :, 1:2].detach())

                output_af = self.af_decoder(torch.cat((mean, sigma, state_ix_af), -1))
            else:
                if self.joint_model_af_training:
                    buckets = output_f.softmax(-1)
                    x_in = x_emb[:, single_eval_pos:]
                else:
                    buckets = output_f.detach().softmax(-1)
                    x_in = x_emb[:, single_eval_pos:].detach()
                output_af = self.af_decoder(torch.cat((buckets, x_in, state_ix_af), -1))

        elif self.af_name == "none":
            output_af = output_f


        # entropy correction with temperature
        output_af = output_af / self.temperature
        # output_af = torch.ones_like(output_af)

        if self.use_value_network:
            output_vf = self.value_head(state_ix)
            return output_af, output_f, output_vf

        return output_af, output_f, None