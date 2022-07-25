# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import RelaxedOneHotCategorical
from torch.utils.data import DataLoader, TensorDataset

from .deep_ensemble import BaseNet, DeepEnsemble
from ..util import construct_hidden

class GumbelSelectionLayer(nn.Module):
    def __init__(self, in_features, out_features, temperature = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.logits       = nn.Parameter(torch.zeros(out_features, in_features))
        self.temperature  = temperature

    @property
    def dist(self):
        return RelaxedOneHotCategorical(temperature = self.temperature, logits = self.logits)

    def forward(self, x):
        # TODO: treat training and eval seperately
        w   = self.dist.rsample()
        out = F.linear(x, weight = w)
        return out

    def extra_repr(self):
        return f'in_features = {self.in_features}, out_features = {self.out_features}'

class GumbelNet(BaseNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)

        # NOTE: here we only apply feature selection to continuous parameters
        # for faster implementation, as you need to deal with enum transformer
        # (embedding or onehot), it's not hard but requires some more work... 
        self.reduced_dim    = conf.setdefault('reduced_dim', num_cont // 2 + 1)
        self.feature_select = GumbelSelectionLayer(self.num_cont, self.reduced_dim)
        if self.num_cont > 0:
            # NOTE: we still keep `feature_select` even if there's not numeric
            # parameter, because in `fit_one`, the temperature is still
            # annealed
            self.eff_dim = (self.eff_dim - self.num_cont) + self.reduced_dim
            self.hidden  = construct_hidden(self.eff_dim, self.num_layers, self.num_hiddens)
    
    def forward(self, xc,  xe):
        if self.num_cont > 0:
            xc = self.feature_select(xc)
        return super().forward(xc, xe)

class GumbelDeepEnsemble(DeepEnsemble):
    support_grad = False
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.basenet_cls = GumbelNet

    def fit_one(self, Xc, Xe, y, idx):
        torch.seed()
        dataset   = TensorDataset(Xc, Xe, y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        if self.models is not None and len(self.models) == self.num_ensembles:
            model = deepcopy(self.models[idx])
        else:
            model = self.basenet_cls(self.num_cont, self.num_enum, self.num_out, **self.conf)
        opt = torch.optim.Adam(model.parameters(), lr = self.lr)
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss  = 0
            temperature = 0.8**(epoch) + 0.1   
            model.feature_select.temperature = temperature
            for bxc, bxe, by in loader:
                py        = model(bxc, bxe)
                data_loss = self.loss(py, by)
                reg_loss  = 0.
                for n, p in model.named_parameters():
                    if 'weight' in n:
                        reg_loss += self.l1 * p.abs().sum() / (y.shape[0] * y.shape[1])
                mask_loss = 0.
                loss = data_loss + reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += data_loss * bxc.shape[0]
            if epoch % self.print_every == 0:
                if self.verbose:
                    print("Epoch %d, %s loss = %g, temperature = %.3f" % (epoch, self.loss_name, epoch_loss / Xc.shape[0], temperature), flush = True)
        model.eval()
        return model
