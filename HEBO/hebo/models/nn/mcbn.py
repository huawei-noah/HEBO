# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, FloatTensor, LongTensor

from .deep_ensemble import BaseNet, DeepEnsemble
from ..util import construct_hidden

class MLPBN(BaseNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.bn_before_act = conf.get('bn_before_act', True)
        self.hidden = construct_hidden(self.eff_dim, self.num_layers, self.num_hiddens)
        net_list    = list(self.hidden)

        if self.bn_before_act:
            net_list = sum([[ele] if isinstance(ele, nn.Linear) else [nn.BatchNorm1d(self.num_hiddens), ele] for ele in net_list[:-2]], []) + net_list[-2:]
        else:
            net_list = sum([[ele] if isinstance(ele, nn.Linear) else [ele, nn.BatchNorm1d(self.num_hiddens)] for ele in net_list[:-2]], []) + net_list[-2:]
        self.hidden = nn.Sequential(*net_list)


class MCBNEnsemble(DeepEnsemble):
    """
    Implement the idea of 'Bayesian Uncertainty Estimation for Batch Normalized Deep Networks'
    http://proceedings.mlr.press/v80/teye18a/teye18a.pdf
    """
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.basenet_cls    = MLPBN
        self.inf_batch_size = self.conf.get('inf_batch_size', self.batch_size)

    def fit_one(self, Xc, Xe, y, idx, **fitting_conf):
        model = super().fit_one(Xc, Xe, y, idx, **fitting_conf)
        model.train()

        # XXX: Using a batch of training data to set statistics of BN layer
        num_data  = y.shape[0]
        batch_idx = np.random.choice(num_data, self.inf_batch_size)
        bxc       = Xc[batch_idx]
        bxe       = Xe[batch_idx]
        for layer in model.hidden:
            if isinstance(layer, nn.BatchNorm1d):
                layer.momentum = 1.0
        model(bxc, bxe)
        model.eval()
        return model
