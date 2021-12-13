# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, out_features, dropout=0.05, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout    = nn.Dropout(p=dropout)

        position    = torch.arange(0, max_len, 1, dtype=torch.float).unsqueeze(1)
        dividers    = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))

        pe          = torch.zeros(max_len, out_features)
        pe[:, 0::2] = torch.sin(position * dividers)
        pe[:, 1::2] = torch.cos(position * dividers)
        pe          = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x:      [sequence length, batch size, out_features]
        output: [sequence length, batch size, out_features]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Embedding):
    def __init__(self, out_features, dropout=0.05, max_len=5000):
        super().__init__(max_len, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight  = self.weight.data.unsqueeze(1)
        x       = x + weight[:x.size(0), :]
        return self.dropout(x)