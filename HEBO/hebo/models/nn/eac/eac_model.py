# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import abstractmethod
import math

import torch
from torch import FloatTensor, LongTensor
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from hebo.models.nn.eac.embedding_alignment_cell import EmbeddingAlignmentCells as EACs
from hebo.models.nn.eac.positional_encoding import LearnablePositionalEncoding as LPE
from hebo.models.util import construct_hidden

class EACNet(nn.Module):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        """
        Base model from which EAC-MLP/EAC-RNN/EAC-LSTM/EAC-Transformer is built.
        TODO:
            - random prior
        """
        super(EACNet, self).__init__()
        self.num_cont       = num_cont      # the number of continuous/numerical feature
        self.num_enum       = num_enum      # the number of enumerate feature
        self.num_out        = num_out       # the number of output
        self.conf           = conf          # key word arguments
        self.stages         = conf.get('stages',        [])
        self.space          = conf.get('space',         None)
        self.output_noise   = conf.get('output_noise',  True)   # flag for noisy model
        self.noise_lb       = conf.get('noise_lb',      1e-4)
        self.model_type     = conf.get('model_type',    'rnn')

        # EAC
        self.out_features   = conf.get('out_features',   64)
        self.nhidden_eac    = conf.get('nhidden_eac',    64)
        self.nlayer_eac     = conf.get('nlayer_eac',     1)
        self.enum_trans     = conf.get('enum_trans',     'onehot')
        self.share_weights  = conf.get('share_weights',  False)
        self.eac            = EACs(stages=self.stages,
                                   space=self.space,
                                   out_features=self.out_features,
                                   num_hiddens=self.nhidden_eac,
                                   num_layers=self.nlayer_eac,
                                   enum_trans=self.enum_trans,
                                   share_weights=self.share_weights)

        # output layer
        self.mu = nn.Linear(self.out_features, self.num_out)
        if self.output_noise:
            self.sigma2 = nn.Sequential(
                nn.Linear(self.out_features, self.num_out),
                nn.Softplus())

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        pass


class EACMLP(EACNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super(EACMLP, self).__init__(num_cont, num_enum, num_out, **conf)
        self.num_layers = self.conf.get('nlayer_mlp', 2)
        self.hidden = construct_hidden(dim=self.out_features,
                                       num_layers=self.num_layers,
                                       num_hiddens=self.out_features)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            else:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        inp = self.eac(Xc, Xe)
        out = torch.sum(inp, dim=0)
        out = self.hidden(out)

        mu  = self.mu(out)
        if not self.output_noise:
            return mu
        else:
            sigma2  = self.sigma2(out)
            return torch.cat((mu, sigma2 + self.noise_lb), dim=1)


class EACRNN(EACNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super(EACRNN, self).__init__(num_cont, num_enum, num_out, **conf)
        self.rnn_layer  = nn.RNN if self.model_type == 'rnn' else nn.LSTM
        self.num_layers = self.conf.get('nlayer_rnn', 1)
        self.rnn        = self.rnn_layer(input_size=self.out_features,
                                         hidden_size=self.out_features,
                                         num_layers=self.num_layers,
                                         batch_first=False)
        self.init_weights()

    @property
    def initHidden(self):
        # return torch.zeros(self.num_layers, 1, self.hidden_size)
        return None

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                nn.init.zeros_(p)
            else:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        """
        :param Xc:  (batch_size, dim0)
        :param Xe:  (batch_size, dim1)
        :return: mu or (mu, sigma2)
        """
        inp         = self.eac(Xc, Xe)
        out, hidden = self.rnn(inp, self.initHidden)
        out         = out[-1, :, :]

        mu  = self.mu(out)
        if not self.output_noise:
            return mu
        else:
            sigma2  = self.sigma2(out)
            return torch.cat((mu, sigma2 + self.noise_lb), dim=1)


class EACTransformerEncoder(EACNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super(EACTransformerEncoder, self).__init__(num_cont, num_enum, num_out, **conf)
        self.dropout    = self.conf.get('dropout',      0.1)
        self.nhead      = self.conf.get('nhead',        8)
        self.num_layers = self.conf.get('nlayer_tran',  6)

        # transformer encoder
        self.src_mask       = None
        # self.pos_encoder    = PositionalEncoding(out_features=self.out_features,
        #                                          dropout=self.dropout)
        self.pos_encoder    = LPE(out_features=self.out_features,
                                  dropout=self.dropout)
        encoder_layers      = TransformerEncoderLayer(d_model=self.out_features,
                                                      nhead=self.nhead,
                                                      dropout=self.dropout,
                                                      dim_feedforward=512,
                                                      activation="relu")
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=self.num_layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for n, p in self.named_parameters():
            if 'bias' in n:
                nn.init.zeros_(p)
            else:
                nn.init.uniform_(p, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).\
            masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        """
        :param Xc:
        :param Xe:
        :return: mu or (mu, sigma2)
        """
        src     = self.eac(Xc, Xe) # (seq_len, batch_size, dim)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = self._generate_square_subsequent_mask(len(src))

        src    *= math.sqrt(self.out_features)    # why?
        src     = self.pos_encoder(src)
        out     = self.transformer_encoder(src, self.src_mask)
        out     = torch.mean(out, dim=0)

        mu  = self.mu(out)
        if not self.output_noise:
            return mu
        else:
            sigma2  = self.sigma2(out)
            return torch.cat((mu, sigma2 + self.noise_lb), dim=1)
