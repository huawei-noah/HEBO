# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbTransform(nn.Module):
    def __init__(self, num_uniqs, **conf):
        super().__init__()
        self.emb_sizes = conf.get('emb_sizes')
        if self.emb_sizes is None:
            self.emb_sizes = [min(50, 1 + v // 2) for v in num_uniqs]
        
        self.emb = nn.ModuleList([])
        for num_uniq, emb_size in zip(num_uniqs, self.emb_sizes):
            self.emb.append(nn.Embedding(num_uniq, emb_size))

    @property
    def num_out_list(self) -> [int]:
        return self.emb_sizes
    
    @property
    def num_out(self)->int:
        return sum(self.emb_sizes)

    def forward(self, xe):
        return torch.cat([self.emb[i](xe[:, i]).view(xe.shape[0], -1) for i in range(len(self.emb))], dim = 1)

class OneHotTransform(nn.Module):
    def __init__(self, num_uniqs):
        super().__init__()
        self.num_uniqs = num_uniqs

    @property
    def num_out_list(self) -> [int]:
        return self.num_uniqs

    @property
    def num_out(self)->int:
        return sum(self.num_uniqs)

    def forward(self, xe):
        return torch.cat([F.one_hot(xe[:, i], self.num_uniqs[i]) for i in range(xe.shape[1])], dim = 1).float()
