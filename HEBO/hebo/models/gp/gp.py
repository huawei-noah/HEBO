# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import pickle

from torch import Tensor, FloatTensor, LongTensor
from pathlib import Path
from gpytorch.priors  import GammaPrior
from gpytorch.priors.torch_priors import LogNormalPrior
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.constraints import Interval, GreaterThan
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..util import filter_nan
from ..base_model import BaseModel
from ..layers import EmbTransform
from ..scalers import TorchMinMaxScaler, TorchStandardScaler

class GP(BaseModel):
    support_grad = True
    support_multi_output = True
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.lr           = conf.get('lr', 3e-2)
        self.num_epochs   = conf.get('num_epochs', 100)
        self.verbose = conf.get('verbose', False)
        self.print_every  = conf.get('print_every', 10)
        self.noise_free   = conf.get('noise_free', False)
        self.pred_likeli  = conf.get('pred_likeli', True)
        self.noise_lb     = conf.get('noise_lb', 1e-5)
        self.xscaler      = TorchMinMaxScaler((-1, 1))
        self.yscaler      = TorchStandardScaler()

    def fit_scaler(self, Xc : Tensor, Xe : Tensor, y : Tensor):
        if Xc is not None and Xc.shape[1] > 0:
            self.xscaler.fit(Xc)
        self.yscaler.fit(y)
    
    def xtrans(self, Xc : Tensor, Xe : Tensor, y : Tensor = None):
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

    def fit(self, Xc : Tensor, Xe : Tensor, y : Tensor):
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'all')
        self.fit_scaler(Xc, Xe, y)
        Xc, Xe, y = self.xtrans(Xc, Xe, y)

        assert(Xc.shape[1] == self.num_cont)
        assert(Xe.shape[1] == self.num_enum)
        assert(y.shape[1]  == self.num_out)

        self.Xc = Xc
        self.Xe = Xe
        self.y  = y

        n_constr = GreaterThan(self.noise_lb)
        n_prior  = LogNormalPrior(-4.63, 0.5)
        if self.num_out == 1:
            self.lik = GaussianLikelihood(noise_constraint = n_constr, noise_prior = n_prior)
        else:
            self.lik = MultitaskGaussianLikelihood(num_tasks = self.num_out, noise_constraint = n_constr, noise_prior = n_prior)
        self.gp = GPyTorchModel(self.Xc, self.Xe, self.y, self.lik, **self.conf)

        if self.num_out == 1: # XXX: only tuned for single-output BO
            if self.num_cont > 0:
                self.gp.kern.outputscale = self.y.var()
                lscales = self.gp.kern.base_kernel.lengthscale.detach().clone().view(1, -1)
                for i in range(self.num_cont):
                    lscales[0, i] = torch.pdist(self.Xc[:, i].view(-1, 1)).median().clamp(min = 0.02)
                self.gp.kern.base_kernel.lengthscale = lscales
            if self.noise_free:
                self.gp.likelihood.noise  = self.noise_lb * 1.1
                self.gp.likelihood.raw_noise.requires_grad = False
            else:
                self.gp.likelihood.noise  = max(1e-2, self.noise_lb)

        self.gp.train()
        self.lik.train()

        opt = torch.optim.LBFGS(self.gp.parameters(), lr = self.lr, max_iter = 5, line_search_fn = 'strong_wolfe')
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.lik, self.gp)
        for epoch in range(self.num_epochs):
            def closure():
                dist = self.gp(self.Xc, self.Xe)
                loss = -1 * mll(dist, self.y.squeeze())
                opt.zero_grad()
                loss.backward()
                return loss
            opt.step(closure)
            if self.verbose and ((epoch + 1) % self.print_every == 0 or epoch == 0):
                print('After %d epochs, loss = %g' % (epoch + 1, closure().item()), flush = True)
        self.gp.eval()
        self.lik.eval()

    def predict(self, Xc, Xe):
        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            pred = self.gp(Xc, Xe)
            if self.pred_likeli:
                pred = self.lik(pred)
            mu_  = pred.mean.reshape(-1, self.num_out)
            var_ = pred.variance.reshape(-1, self.num_out)
        mu  = self.yscaler.inverse_transform(mu_)
        var = var_ * self.yscaler.std**2
        return mu, var.clamp(min = torch.finfo(var.dtype).eps)

    def sample_y(self, Xc, Xe, n_samples = 1) -> FloatTensor:
        """
        Should return (n_samples, Xc.shape[0], self.num_out) 
        """
        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.debug(False):
            if self.pred_likeli:
                pred = self.lik(self.gp(Xc, Xe))
            else:
                pred = self.gp(Xc, Xe)
            samp = pred.rsample(torch.Size((n_samples,))).view(n_samples, Xc.shape[0], self.num_out)
            return self.yscaler.inverse_transform(samp)

    def sample_f(self):
        raise NotImplementedError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def noise(self):
        if self.num_out == 1:
            return (self.gp.likelihood.noise * self.yscaler.std**2).view(self.num_out).detach()
        else:
            return (self.gp.likelihood.task_noises * self.yscaler.std**2).view(self.num_out).detach()

class GPyTorchModel(gpytorch.models.ExactGP):
    def __init__(self, 
            x   : torch.Tensor,
            xe  : torch.Tensor,
            y   : torch.Tensor,
            lik : GaussianLikelihood, 
            **conf):
        super().__init__((x, xe), y.squeeze(), lik)
        mean     = conf.get('mean', ConstantMean())
        kern     = conf.get('kern', ScaleKernel(MaternKernel(nu = 1.5, ard_num_dims = x.shape[1]), outputscale_prior = GammaPrior(0.5, 0.5)))
        kern_emb = conf.get('kern_emb', MaternKernel(nu = 2.5))

        self.multi_task = y.shape[1] > 1
        self.mean  = mean if not self.multi_task else MultitaskMean(mean, num_tasks = y.shape[1])
        if x.shape[1] > 0:
            self.kern = kern if not self.multi_task else MultitaskKernel(kern, num_tasks = y.shape[1])
        if xe.shape[1] > 0:
            assert 'num_uniqs' in conf
            num_uniqs = conf['num_uniqs']
            emb_sizes = conf.get('emb_sizes', None)
            self.emb_trans = EmbTransform(num_uniqs, emb_sizes = emb_sizes)
            self.kern_emb  = kern_emb if not self.multi_task else MultitaskKernel(kern_emb, num_tasks = y.shape[1])

    def forward(self, x, xe):
        m = self.mean(x)
        if x.shape[1] > 0:
            K = self.kern(x)
            if xe.shape[1] > 0:
                x_emb  = self.emb_trans(xe)
                K     *= self.kern_emb(x_emb)
        else:
            K = self.kern_emb(self.emb_trans(xe))
        return MultivariateNormal(m, K) if not self.multi_task else MultitaskMultivariateNormal(m, K)
