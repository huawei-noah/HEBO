from model.base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
from einops.layers.torch import Rearrange, Repeat

class EncoderCDR3(Module):
    def __init__(self):
        super(EncoderCDR3, self).__init__()
        self.encoder = nn.Sequntial(
            Rearrange('b n d -> b (n d)'),
            nn.Linear(220, 300),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(100, 40),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
        )
        self.q_mu = nn.Linear(40, 40)
        self.q_logsigma = nn.Linear(40, 40)

    def cat_to_onehot(self, x):
        x_onehot = torch.zeros(x.shape[0], 20).to(x.device)
        x_onehot.scatter_(1, x, 1.0)
        return x_onehot

    def forward(self, x):
        x = self.cat_to_onehot(x)
        return self.encoder(x)

class DecoderCDR3(Module):
    def __init__(self):
        super(DecoderCDR3, self).__init__()
        self.decoder = nn.Sequntial(
            nn.Linear(40, 100),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(100, 300),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.Linear(300, 220),
            nn.BatchNorm1d(),
            Rearrange('b (n d) -> b n d', n=11, d=20),
        )

    def forward(self, x):
        return self.decoder(x)

