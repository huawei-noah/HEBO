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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class SGLD(torch.optim.SGD):
    def __init__(self, params, lr, factor = 1., pretrain_step = 0, **kw_args):
        """
        factor: if the loss used is something like nn.MSELoss(reduction = 'mean'), then factor should be 1. / N, where N is the number of data
        pretrain_step: number of steps that only uses SGD to optimize the objective
        """
        super().__init__(params, lr, **kw_args)
        self.factor        = factor
        self.pretrain_step = pretrain_step
        self.n_step        = 0

    @torch.no_grad()
    def step(self, **kw_args):
        super().step(**kw_args)
        self.n_step += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.n_step > self.pretrain_step:
                    lr         = group['lr']
                    noise_var  = 2. * lr
                    noise_std  = np.sqrt(noise_var)
                    p.add_(self.factor * noise_std * torch.randn_like(p))

class pSGLD(torch.optim.RMSprop):
    def __init__(self, params, factor = 1., pretrain_step = 0, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)
        self.factor        = factor
        self.pretrain_step = pretrain_step
        self.n_step        = 0

    @torch.no_grad()
    def step(self, closure = None):
        super().step(closure)
        self.n_step += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.n_step > self.pretrain_step:
                    square_avg  = self.state[p]['square_avg']
                    lr          = group['lr']
                    eps         = group['eps']
                    avg         = square_avg.sqrt().add_(eps)
                    noise_var   = 2 * lr / avg
                    p.add_(self.factor * noise_var.sqrt() * torch.randn_like(p))

class pSGLDEnsemble(DeepEnsemble):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.factor          = conf.get('factor')
        self.pretrain_epochs = conf.get('pretrain_epochs', 50)

    def fit_one(self, Xc, Xe, y, idx, **fitting_conf):
        torch.seed()
        dataset   = TensorDataset(Xc, Xe, y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, drop_last = y.shape[0] > self.batch_size)
        if self.models is not None and len(self.models) == self.num_ensembles:
            model = deepcopy(self.models[idx])
        else:
            model = self.basenet_cls(self.num_cont, self.num_enum, self.num_out, **self.conf)

        factor = 1. / y.numel() if self.factor is None else self.factor
        opt = pSGLD(model.parameters(),
                lr            = self.lr,
                factor        = factor,
                pretrain_step = self.pretrain_epochs * y.shape[0] // self.batch_size)

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
        model.eval()
        return model
