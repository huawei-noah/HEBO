# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn as nn
import gpytorch

from copy import deepcopy
from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior
from gpytorch.priors.torch_priors import LogNormalPrior
from gpytorch.kernels import ScaleKernel, MaternKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, NaturalVariationalDistribution

from ..util import filter_nan
from ..base_model import BaseModel
from ..layers import EmbTransform
from ..scalers import TorchMinMaxScaler, TorchStandardScaler
from .gp import DummyFeatureExtractor, default_kern


class SVGPLayer(ApproximateGP):
    def __init__(self, mean, kern, u, learn_u, use_ngd):
        num_inducing = u.shape[0]
        if use_ngd:
            variational_distribution = NaturalVariationalDistribution(num_inducing)
        else:
            variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(self, u, variational_distribution, learn_inducing_locations = learn_u)
        super().__init__(variational_strategy)

        self.mean = mean
        self.cov  = kern

    def forward(self, x_all):
        m = self.mean(x_all)
        K = self.cov(x_all)
        return MultivariateNormal(m, K)

class SVGPList(gpytorch.Module):
    def __init__(self, gp_list):
        super().__init__()
        self.num_out = len(gp_list)
        self.gp      = nn.ModuleList(gp_list)

    def __getitem__(self, i):
        return self.gp[i]
    
    def forward(self, x_all, y):
        if self.training:
            return [self.gp[i](x_all) for i in range(self.num_out)]
        else:
            dist_list = []
            for i in range(self.num_out):
                valid_idx = torch.isfinite(y[:, i])
                dist_list.append(self.gp[i](x_all[valid_id]))
            return dist_list

class SVGPModel(gpytorch.Module):
    def __init__(self, x, xe, y, num_inducing = 128, **conf):
        super().__init__()
        self.num_out = y.shape[1]
        self.fe = deepcopy(conf.get('fe', DummyFeatureExtractor(x.shape[1], xe.shape[1], conf.get('num_uniqs'), conf.get('emb_sizes'))))
        self.gp = SVGPList([
            SVGPLayer(
                mean    = deepcopy(conf.get('mean', ConstantMean())), 
                kern    = deepcopy(conf.get('kern', default_kern(x, xe, y[:, i], self.fe.total_dim, conf.get('ard_kernel', True), conf.get('fe')))),
                u       = self.init_u(x, xe, num_inducing),
                learn_u = conf.get('learn_u', True), 
                use_ngd = conf.get('use_ngd', False), 
                )
            for i in range(self.num_out)
        ])

    def forward(self, x, xe, y = None):
        x_all = self.fe(x, xe)
        if not self.training:
            return [self.gp[i](x_all) for i in range(self.num_out)]
        else:
            dist_list = []
            for i in range(self.num_out): # for multi-task GP, there might be a lot of missing values
                valid_idx = torch.isfinite(y[:, i])
                dist_list.append(self.gp[i](x_all[valid_idx]))
            return dist_list

    def init_u(self, x, xe, num_inducing):
        num_data     = x.shape[0]
        num_inducing = min(num_inducing, num_data)
        u_idx        = np.random.choice(num_data, num_inducing, replace = False)
        with torch.no_grad():
            u = self.fe(x[u_idx], xe[u_idx]).detach().clone()
            return u


class SVGP(BaseModel):
    support_grad = True
    support_multi_output = True
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)

        self.use_ngd      = conf.get('use_ngd', False)
        self.lr           = conf.get('lr',    1e-2)
        self.lr_vp        = conf.get('lr_vp', 1e-1)
        self.lr_fe        = conf.get('lr_fe', 1e-3)
        self.num_inducing = conf.get('num_inducing', 128)
        self.ard_kernel   = conf.get('ard_kernel', True)
        self.pred_likeli  = conf.get('pred_likeli', True) # whether or not consider noise to the uncertainty
        self.beta         = conf.get('beta', 1.0)

        self.batch_size   = conf.get('batch_size', 64)
        self.num_epochs   = conf.get('num_epochs', 300)
        self.verbose      = conf.get('verbose', False)
        self.print_every  = conf.get('print_every', 10)
        self.noise_lb     = conf.get('noise_lb', 1e-5)
        self.xscaler      = TorchMinMaxScaler((-1, 1))
        self.yscaler      = TorchStandardScaler()

    def fit_scaler(self, Xc : FloatTensor, Xe : LongTensor, y : FloatTensor):
        if Xc is not None and Xc.shape[1] > 0:
            self.xscaler.fit(Xc)
        self.yscaler.fit(y)
    
    def xtrans(self, Xc : FloatTensor, Xe : LongTensor, y : FloatTensor = None):
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

    def fit(self, Xc : FloatTensor, Xe : LongTensor, y : FloatTensor):
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'any')
        self.fit_scaler(Xc, Xe, y)
        Xc, Xe, y = self.xtrans(Xc, Xe, y)

        assert(Xc.shape[1] == self.num_cont)
        assert(Xe.shape[1] == self.num_enum)
        assert(y.shape[1]  == self.num_out)

        n_constr = GreaterThan(self.noise_lb)
        self.gp  = SVGPModel(Xc, Xe, y, **self.conf)
        self.lik = nn.ModuleList([GaussianLikelihood(noise_constraint = n_constr) for _ in range(self.num_out)])

        self.gp.train()
        self.lik.train()

        ds  = TensorDataset(Xc, Xe, y)
        dl  = DataLoader(ds, batch_size = self.batch_size, shuffle = True, drop_last = y.shape[0] > self.batch_size)
        if self.use_ngd:
            opt = torch.optim.Adam([
                {'params' : self.gp.fe.parameters(), 'lr' : self.lr_fe}, 
                {'params' : self.gp.gp.hyperparameters()}, 
                {'params' : self.lik.parameters()},
                ], lr = self.lr)
            opt_ng = gpytorch.optim.NGD(self.gp.variational_parameters(), lr = self.lr_vp, num_data = y.shape[0])
        else:
            opt = torch.optim.Adam([
                {'params' : self.gp.fe.parameters(), 'lr' : self.lr_fe}, 
                {'params' : self.gp.gp.hyperparameters()}, 
                {'params' : self.gp.gp.variational_parameters(), 'lr' : self.lr_vp}, 
                {'params' : self.lik.parameters()},
                ], lr = self.lr)


        mll = [gpytorch.mlls.VariationalELBO(self.lik[i], self.gp.gp[i], num_data = y.shape[0], beta = self.beta) for i in range(self.num_out)]
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            epoch_cnt  = 1e-6
            for bxc, bxe, by in dl:
                dist_list = self.gp(bxc, bxe, by)
                loss      = 0
                valid     = torch.isfinite(by)
                for i, dist in enumerate(dist_list):
                    loss += -1 * mll[i](dist, by[valid[:, i], i]) * valid[:, i].sum()
                loss /= by.shape[0]

                if self.use_ngd:
                    opt.zero_grad()
                    opt_ng.zero_grad()
                    loss.backward()
                    opt.step()
                    opt_ng.step()
                else:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                epoch_loss += loss.item()
                epoch_cnt  += 1
            epoch_loss /= epoch_cnt
            if self.verbose and ((epoch + 1) % self.print_every == 0 or epoch == 0):
                print('After %d epochs, loss = %g' % (epoch + 1, epoch_loss), flush = True)
        self.gp.eval()
        self.lik.eval()

    def predict(self, Xc, Xe):
        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            pred = self.gp(Xc, Xe)
            if self.pred_likeli:
                for i in range(self.num_out):
                    pred[i] = self.lik[i](pred[i])
            mu_  = torch.cat([pred[i].mean.reshape(-1, 1) for i in range(self.num_out)], dim = 1)
            var_ = torch.cat([pred[i].variance.reshape(-1, 1) for i in range(self.num_out)], dim = 1)
        mu  = self.yscaler.inverse_transform(mu_)
        var = var_ * self.yscaler.std**2
        return mu, var.clamp(min = torch.finfo(var.dtype).eps)

    def sample_y(self, Xc, Xe, n_samples = 1) -> FloatTensor:
        """
        Should return (n_samples, Xc.shape[0], self.num_out) 
        """
        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.debug(False):
            pred = self.gp(Xc, Xe)
            if self.pred_likeli:
                for i in range(self.num_out):
                    pred[i] = self.lik[i](pred[i])
            samp = [pred[i].rsample(torch.Size((n_samples, ))).reshape(n_samples, -1, 1) for i in range(self.num_out)]
            samp = torch.cat(samp, dim = -1)
            return self.yscaler.inverse_transform(samp)

    def sample_f(self):
        raise NotImplementedError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def noise(self):
        noise = torch.FloatTensor([lik.noise for lik in self.lik]).view(self.num_out).detach()
        return noise * self.yscaler.std**2
