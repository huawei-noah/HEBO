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
from copy import deepcopy

from .deep_ensemble import BaseNet, DeepEnsemble
from .fe_layers import get_fe_layer, StochasticGate

class FeNet(BaseNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.fe_layer       = get_fe_layer(conf.get('fe_layer', 'stg'))
        self.temperature    = conf.get('temperature')
        if self.temperature:
            self.feature_select = self.fe_layer(self.eff_dim, self.temperature)
        else:
            self.feature_select = self.fe_layer(self.eff_dim)

    def forward(self, xc,  xe):
        # NOTE: we ignore random prior network, as it breaks irrelavent features
        inputs = self.feature_select(self.xtrans(xc, xe))
        hidden = self.hidden(inputs)
        mu     = self.mu(hidden)
        out    = torch.cat((mu, self.noise_lb + self.sigma2(hidden)), dim = 1) if self.output_noise else mu
        return out

class FeDeepEnsemble(DeepEnsemble):
    support_grad = False
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.basenet_cls = FeNet
        self.mask_reg    = conf.get('mask_reg', 0.1)
        self.end_temp    = conf.get('end_temp', 0.1)
        self.start_temp  = conf.get('start_temp', 1.0)
        self.anneal_base = conf.get('anneal_base', 0.99)

    def fit_one(self, Xc, Xe, y, idx):
        torch.seed()
        dataset   = TensorDataset(Xc, Xe, y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, drop_last = y.shape[0] > self.batch_size)
        if self.models is not None and len(self.models) == self.num_ensembles:
            model = deepcopy(self.models[idx])
        else:
            model = self.basenet_cls(self.num_cont, self.num_enum, self.num_out, **self.conf)
        if idx == 0 and self.verbose:
            print(model, flush = True)
        opt = torch.optim.Adam(model.parameters(), lr = self.lr)
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss  = 0
            if not isinstance(model.feature_select, StochasticGate):
                temperature = torch.tensor(self.start_temp * self.anneal_base **epoch).clamp(self.end_temp)
                model.feature_select.temperature = temperature
            for bxc, bxe, by in loader:
                py        = model(bxc, bxe)
                data_loss = self.loss(py, by)
                reg_loss  = 0.
                for n, p in model.named_parameters():
                    if 'weight' in n:
                        reg_loss += self.l1 * p.abs().sum() / (y.shape[0] * y.shape[1])
                mask_loss = 0.
                if Xc.shape[1] > 0:
                    mask_loss = self.mask_reg * model.feature_select.mask_norm / (y.shape[0] * y.shape[1])
                loss = data_loss + reg_loss + mask_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += data_loss * bxc.shape[0]
            if epoch % self.print_every == 0:
                if self.verbose:
                    print("Epoch %d, %s loss = %g, temperature = %.3f, mask_norm = %.3f" % (epoch, self.loss_name, epoch_loss / Xc.shape[0], model.feature_select.temperature, model.feature_select.mask_norm), flush = True)
        model.eval()
        return model
