# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import sys

import traceback
import numpy as np
import torch
import torch.nn as nn

from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, FloatTensor, LongTensor
from copy import deepcopy

from ..base_model import BaseModel
from ..layers import EmbTransform
from ..scalers import TorchMinMaxScaler, TorchStandardScaler

class DeepEnsemble(BaseModel):
    support_ts           = True
    support_grad         = True
    support_multi_output = True
    support_warm_start   = True
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.bootstrap     = conf.get('bootstrap',     False)
        self.rand_prior    = conf.get('rand_prior',    False)
        self.output_noise  = conf.get('output_noise',  True)
        self.num_ensembles = conf.get('num_ensembles', 5)
        self.num_process   = conf.get('num_processes', 1)
        self.num_epochs    = conf.get('num_epochs',    500)
        self.print_every   = conf.get('print_every',   50)

        self.num_layers    = conf.get('num_layers',    1)
        self.num_hiddens   = conf.get('num_hiddens',   128)
        self.l1            = conf.get('l1',            1e-3)
        self.batch_size    = conf.get('batch_size',    32)
        self.lr            = conf.get('lr',            5e-3)
        self.adv_eps       = conf.get('adv_eps',       0.)
        self.verbose       = conf.get('verbose',       False)
        assert self.num_ensembles > 0

        self.xscaler = TorchMinMaxScaler((-1, 1))
        self.yscaler = TorchStandardScaler()

        self.loss       = self.loss_likelihood if self.output_noise else self.loss_mse
        self.loss_name  = "NLL" if self.output_noise else "MSE"
        self.models     = None
        self.sample_idx = 0
        self.noise_est  = torch.zeros(self.num_out)

    @property
    def fitted(self):
        return self.models is not None

    @property
    def noise(self) -> FloatTensor:
        return self.noise_est

    def fit(self, Xc_ : FloatTensor, Xe_ : LongTensor, y_ : FloatTensor):
        valid = torch.isfinite(y_).any(dim = 1)
        Xc    = Xc_[valid] if Xc_ is not None else None
        Xe    = Xe_[valid] if Xe_ is not None else None
        y     = y_[valid]

        if self.models is None:
            self.fit_scaler(Xc, Xe, y)
        Xc, Xe, y = self.trans(Xc, Xe, y)

        if self.num_process > 1:
            with Pool(self.num_process) as p:
                self.models = p.starmap(self.fit_one, [(Xc.clone(), Xe.clone(), y.clone(), model_idx) for model_idx in range(self.num_ensembles)]) 
        else:
            self.models = [self.fit_one(Xc, Xe, y, i) for i in range(self.num_ensembles)]
    
        assert None not in self.models
        self.sample_idx = 0

        with torch.no_grad():
            py, _ = self.predict(Xc_, Xe_)
            err   = (py - y_)[valid]
            self.noise_est = (err**2).mean(dim = 0).detach().clone()

    def predict(self, Xc_ : FloatTensor, Xe_ : LongTensor) -> (FloatTensor, FloatTensor):
        Xc, Xe = self.trans(Xc_, Xe_)
        preds  = torch.stack([self.models[i](Xc, Xe) for i in range(self.num_ensembles)])
        if not self.output_noise:
            py  = preds.mean(dim = 0)
            ps2 = 1e-8 + preds.var(dim = 0, unbiased = False) # XXX: var([1.0], unbiased = True) = NaN
        else:
            mu     = preds[:, :, :self.num_out]
            sigma2 = preds[:, :, self.num_out:]
            py     = mu.mean(dim = 0)
            ps2    = mu.var(dim = 0, unbiased = False) + sigma2.mean(axis = 0)
        return self.yscaler.inverse_transform(py), ps2 * self.yscaler.std**2

    def sample_f(self):
        assert(self.fitted)
        idx = self.sample_idx
        self.sample_idx = (self.sample_idx + 1) % self.num_ensembles
        def f(Xc : FloatTensor, Xe : LongTensor) -> FloatTensor:
            model  = self.models[idx]
            Xc, Xe = self.trans(Xc, Xe)
            return self.yscaler.inverse_transform(model(Xc, Xe)[:, :self.num_out])
        return f

    def fit_scaler(self, Xc : Tensor, Xe : Tensor, y : Tensor):
        if Xc is not None and Xc.shape[1] > 0:
            self.xscaler.fit(Xc)
        self.yscaler.fit(y)
    
    def trans(self, Xc : Tensor, Xe : Tensor, y : Tensor = None):
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = torch.zeros(Xe.shape[0], 0)

        if Xe is None:
            Xe_t = torch.zeros(Xc.shape[0], 0).long()
        else:
            Xe_t = Xe.long()

        if y is not None:
            y_t = self.yscaler.transform(y)
            return Xc_t, Xe_t, y_t
        else:
            return Xc_t, Xe_t

    def loss_mse(self, pred, target):
        mask = torch.isfinite(target)
        return nn.MSELoss()(pred[mask], target[mask])

    def loss_likelihood(self, pred, target):
        mask   = torch.isfinite(target)
        mu     = pred[:, :self.num_out][mask]
        sigma2 = pred[:, self.num_out:][mask]
        loss   = 0.5 * (target[mask] - mu)**2 / sigma2 + 0.5 * torch.log(sigma2)
        return torch.mean(loss)

    def fit_one(self, Xc, Xe, y, idx):
        torch.seed()
        dataset   = TensorDataset(Xc, Xe, y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        if self.models is not None and len(self.models) == self.num_ensembles:
            model = deepcopy(self.models[idx])
        else:
            model = BaseNet(self.num_cont, self.num_enum, self.num_out,
                    noise_lb     = 1e-4,
                    num_uniqs    = None if self.num_enum == 0 else self.conf['num_uniqs'], 
                    num_layers   = self.num_layers,
                    num_hiddens  = self.num_hiddens,
                    output_noise = self.output_noise,
                    rand_prior   = self.rand_prior)
        opt       = torch.optim.Adam(model.parameters(), lr = self.lr)
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for bxc, bxe, by in loader:
                py        = model(bxc, bxe)
                data_loss = self.loss(py, by)
                reg_loss  = 0.
                for p in model.parameters():
                    reg_loss += self.l1 * p.abs().sum() / (y.shape[0] * y.shape[1])
                loss = data_loss + reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                # XXX: Adversarial training not used

                epoch_loss += data_loss * bxc.shape[0]
            if epoch % self.print_every == 0:
                if self.verbose:
                    print("Epoch %d, %s loss = %g" % (epoch, self.loss_name, epoch_loss / Xc.shape[0]), flush = True)
        return model

class BaseNet(nn.Module):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__()
        self.num_cont     = num_cont
        self.num_enum     = num_enum
        self.num_out      = num_out
        self.noise_lb     = conf.get('noise_lb', 1e-4)
        self.num_layers   = conf.get('num_layers', 1)  # number of hidden layers
        self.num_hiddens  = conf.get('num_hiddens', 128) # number of hidden units
        self.output_noise = conf.get('output_noise', True)
        self.rand_prior   = conf.get('rand_prior', False)
        self.eff_dim      = num_cont
        if self.num_enum > 0:
            assert 'num_uniqs' in conf, "num_uniqs not in algorithm configuration"
            self.emb_trans  = EmbTransform(conf['num_uniqs'])
            self.eff_dim   += self.emb_trans.num_out

        self.hidden = self.construct_hidden()
        self.mu     = nn.Linear(self.num_hiddens, self.num_out)
        if self.output_noise:
            self.sigma2 = nn.Sequential(
                nn.Linear(self.num_hiddens, self.num_out), 
                nn.Softplus())
        if self.rand_prior: # Randomized Prior Functions for Deep Reinforcement Learning, Ian Osband
            self.prior_net = self.construct_hidden()
            self.prior_net.add_module('prior_net_out', nn.Linear(self.num_hiddens, self.num_out))
        for n, p in self.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            else:
                nn.init.xavier_uniform_(p, gain = nn.init.calculate_gain('relu'))

    def construct_hidden(self):
        layers = [nn.Linear(self.eff_dim, self.num_hiddens), nn.ReLU()]
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.num_hiddens, self.num_hiddens))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def xtrans(self, Xc : FloatTensor, Xe : LongTensor) -> FloatTensor:
        Xall     = Xc.clone() if self.num_cont > 0 else torch.zeros(Xe.shape[0], 0)
        if self.num_enum > 0:
            Xall = torch.cat([Xall, self.emb_trans(Xe)], dim = 1)
        return Xall

    def forward(self, Xc : FloatTensor, Xe : LongTensor) -> FloatTensor:
        inputs = self.xtrans(Xc, Xe)
        prior_out = 0.
        if self.rand_prior:
            with torch.no_grad():
                prior_out = self.prior_net(inputs).detach()
        hidden = self.hidden(inputs)
        mu     = self.mu(hidden) + prior_out
        out    = torch.cat((mu, self.noise_lb + self.sigma2(hidden)), dim = 1) if self.output_noise else mu
        return out
