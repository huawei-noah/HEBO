# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from itertools import combinations
from typing import Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from mcbo.models import ModelBase
from mcbo.search_space import SearchSpace
from mcbo.utils.onehot_utils import onehot_encode
from mcbo.utils.training_utils import remove_repeating_samples


class LinRegModel(ModelBase, torch.nn.Module):

    @property
    def noise(self) -> torch.Tensor:
        return torch.zeros(self.num_out, dtype=self.dtype)

    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        return fit_y

    @property
    def name(self) -> str:
        return f"LR ({self.estimator})"

    def __init__(self,
                 search_space: SearchSpace,
                 order: int = 2,
                 estimator: str = 'sparse_horseshoe',
                 a_prior: float = 2.,
                 b_prior: float = 1.,
                 sparse_horseshoe_threshold: float = 0.1,
                 n_gibbs: int = int(1e3),
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):

        assert search_space.num_nominal == search_space.num_params, 'BOCS\' regression model is only applicable to nominal variables'
        assert estimator in ['mle', 'bayes', 'horseshoe', 'sparse_horseshoe']

        super(LinRegModel, self).__init__(search_space=search_space, num_out=1, dtype=dtype, device=device)

        self.a_prior = a_prior
        self.b_prior = b_prior
        self.a_post = None
        self.b_post = None
        self.sparse_horseshoe_threshold = sparse_horseshoe_threshold

        self.alpha_hs = None  # Parameter sample from the horseshoe prior
        self.alpha_mle = None  # Maximum likelihood estimate of the model parameters
        self.alpha_mean = None  # Bayes posterior mean
        self.alpha_cov = None  # Bayes posterior covariance matrix

        self.order = order
        self.idx_zero = []
        self.idx_nnzero = []
        self.nb_gibbs = n_gibbs
        self.estimator = estimator
        self.nb_coeff = None

        self.nb_vars = search_space.num_nominal
        self.lb = torch.tensor(search_space.nominal_lb)
        self.ub = torch.tensor(search_space.nominal_ub)
        self.num_onehot_vars = int((self.ub - self.lb + 1).sum())

    def fit(self, x: torch.FloatTensor, y: torch.FloatTensor, **kwargs):

        if y.ndim == 1:
            y = y.view(-1, 1)

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]

        # Remove repeating data points
        x, y = remove_repeating_samples(x, y)

        self.x = x.clone().to(self.device, self.dtype)
        self._y = y.clone().to(self.device, self.dtype)

        self.fit_y = self.y_to_fit_y(y=self._y)

        onehot_x = onehot_encode(self.x, self.ub, self.num_onehot_vars)

        x_train = LinRegModel._order_effects(onehot_x, order=self.order)

        (nSamps, _) = x_train.shape

        # check if x_train contains columns with zeros or duplicates
        # and find the corresponding indices
        check_zero = (x_train.cpu() == torch.zeros((nSamps, 1))).all(axis=0)
        self.idx_zero = np.where(check_zero == True)[0]
        self.idx_nnzero = np.where(check_zero == False)[0]

        # remove columns that had zeros in x_train
        if len(self.idx_zero):
            x_train = x_train[:, self.idx_nnzero]

        # Append column of ones
        x_train = torch.cat((torch.ones((nSamps, 1)).to(x_train), x_train), axis=1)

        self.nb_coeff = x_train.shape[1]

        if self.estimator == 'bayes':
            self.alpha_mean, self.alpha_cov, self.a_post, self.b_post = self._train_bayes(x_train, self.fit_y, self.a_prior,
                                                                                          self.b_prior)
        elif self.estimator == 'mle':
            self.alpha_mle = self._train_mle(x_train, self.fit_y)

        elif self.estimator == 'horseshoe' or self.estimator == 'sparse_horseshoe':
            self.alpha_hs, _ = self._train_horseshoe(x_train, self.fit_y, self.nb_gibbs, 0, 1, return_all=False)
        else:
            raise NotImplementedError('Estimator is not implemented!')

    def predict(self, x: torch.FloatTensor, **kwargs) -> (torch.FloatTensor, torch.FloatTensor):
        if self.estimator == 'bayes':
            raise NotImplementedError(
                "The posterior distribution is non Gaussian. Look at https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf to implement this.")

        elif self.estimator == 'mle' or self.estimator == 'horseshoe' or self.estimator == 'sparse_horseshoe':
            raise NotImplementedError(
                'Posterior y_test|X_test, y_train, X_train is not implemented for this estimator.')
        else:
            raise NotImplementedError('Estimator is not implemented!')

    def sample_y(self, x: torch.FloatTensor, n_samples: int, alpha: torch.FloatTensor = None) -> torch.FloatTensor:

        samp = torch.zeros((n_samples, x.shape[0])).to(self.device, self.dtype)

        # Encode the inputs and compute all interactions
        x_ = LinRegModel._order_effects(onehot_encode(x.clone(), self.ub, self.num_onehot_vars), self.order)

        # remove columns that had zeros in x_train
        if len(self.idx_zero):
            x_ = x_[:, self.idx_nnzero]

        # Append column of ones
        x_ = torch.cat((torch.ones((len(x_), 1)).to(x_), x_), axis=1).to(self.device, self.dtype)

        sample_alpha = alpha is None

        for i in range(n_samples):
            if sample_alpha:
                alpha = self._sample_alpha()
            samp[i] = (x_ @ alpha).view(-1)

        return samp

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None:
            assert isinstance(device, torch.device)
            self.device = device
            if self.estimator == 'mle' and self.alpha_mle is not None:
                self.alpha_mle = self.alpha_mle.to(device)
            elif self.estimator == 'horseshoe' and self.alpha_hs is not None:
                self.alpha_hs = self.alpha_hs.to(device)
            elif self.estimator == 'sparse_horseshoe' and self.alpha_hs is not None:
                self.alpha_hs = self.alpha_hs.to(device)

        if dtype is not None:
            assert isinstance(dtype, torch.dtype)
            self.dtype = dtype
            if self.estimator == 'mle' and self.alpha_mle is not None:
                self.alpha_mle = self.alpha_mle.to(dtype)
            elif self.estimator == 'horseshoe' and self.alpha_hs is not None:
                self.alpha_hs = self.alpha_hs.to(dtype)
            elif self.estimator == 'sparse_horseshoe' and self.alpha_hs is not None:
                self.alpha_hs = self.alpha_hs.to(dtype)

        return self

    def _train_bayes(self, x, y, a_prior, b_prior):

        n = x.shape[0]

        # set alpha prior parameters
        alpha_mean_prior = (np.pi / 2) / self.nb_coeff * torch.ones((self.nb_coeff, 1)).to(x)
        alpha_cov_prior = torch.eye(self.nb_coeff).to(x)
        alpha_inv_prior = torch.linalg.inv(alpha_cov_prior)

        # compute alpha posterior parameters
        alpha_prec_ = alpha_inv_prior + x.T @ x
        temp = torch.linalg.solve(torch.eye(n).to(x) + x @ alpha_cov_prior @ x.T, x @ alpha_cov_prior)
        alpha_cov = alpha_cov_prior - alpha_cov_prior @ x.T @ temp  # Sherman-Morrison formula
        alpha_cov = (alpha_cov + alpha_cov.T) / 2  # symmetrizing covariance
        alpha_mean = alpha_cov @ (alpha_inv_prior @ alpha_mean_prior + x.T @ y)

        # compute hyper-parameter posterior parameters. Equation 88 and 89 https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        a_post = a_prior + n / 2
        b_post = b_prior + 0.5 * (
                alpha_mean_prior.T @ alpha_inv_prior @ alpha_mean_prior + y.T @ y - alpha_mean.T @ alpha_prec_ @ alpha_mean).item()

        return alpha_mean, alpha_cov, a_post, b_post

    def _train_mle(self, x, y):

        reg_coeff = 1e-6
        alpha_mle = torch.linalg.solve((x.T @ x + reg_coeff * torch.eye(self.nb_coeff).to(x)), x.T @ y)

        return alpha_mle

    def _train_horseshoe(self, x, y, n_samples, burn_in, thin, return_all=True):
        # Implementation of the Bayesian horseshoe linear regression hierarchy.

        # Discard column of ones
        x = x[:, 1:]

        n, p = x.shape

        # Normalize data
        x, _, _, y, muY = standardise(x, y)

        # Return values
        if return_all:
            alpha_all = torch.zeros((p, n_samples)).to(x)
            sigma2_all = torch.zeros((1, n_samples)).to(x)
            tau2_all = torch.zeros((1, n_samples)).to(x)
            beta2_all = torch.zeros((p, n_samples)).to(x)

        # Initial values
        sigma2 = 1.
        beta2 = torch.tensor(np.random.uniform(size=p)).to(x)
        tau2 = 1.
        nu = torch.ones(p).to(x)
        xi = 1.

        # pre-compute X'*X (used with fastmvg_rue)
        XtX = x.T @ x

        # Gibbs sampler
        k = 0
        step = 0
        while k < n_samples:

            # Sample from the conditional posterior distribution
            sigma = np.sqrt(sigma2)
            Sigma_star = tau2 * beta2.diag()
            # Determine best sampler for conditional posterior of beta's
            if (p > n) and (p > 200):
                alpha = fastmvg(x / sigma, y / sigma, sigma2 * Sigma_star)
            else:
                alpha = fastmvg_rue(x / sigma, XtX / sigma2, y / sigma, sigma2 * Sigma_star)

            # Sample sigma2
            e = y - x @ alpha.view(-1, 1)
            shape = (n + p) / 2.
            scale = (e.T @ e / 2.).item() + ((alpha ** 2 / beta2).sum() / tau2 / 2.).item()
            sigma2 = 1. / np.random.gamma(shape, 1. / scale)

            # Sample beta2
            scale = 1. / nu + alpha ** 2. / (2. * tau2 * sigma2)
            beta2 = torch.tensor(1. / np.random.exponential(1. / scale.cpu())).to(x)

            # Sample tau2
            shape = (p + 1.) / 2.
            scale = (1. / xi + (alpha ** 2. / beta2).sum() / (2. * sigma2)).item()
            tau2 = 1. / np.random.gamma(shape, 1. / scale)

            # Sample nu
            scale = 1. + 1. / beta2
            nu = torch.tensor(1. / np.random.exponential(1. / scale.cpu())).to(x)

            # Sample xi
            scale = 1. + 1. / tau2
            xi = 1. / np.random.exponential(1. / scale)

            # Store samples
            step += 1
            if step > burn_in:
                # thinning
                if (step % thin) == 0:
                    if return_all:
                        alpha_all[:, k] = alpha
                        sigma2_all[:, k] = sigma2
                        tau2_all[:, k] = tau2
                        beta2_all[:, k] = beta2
                    k = k + 1

        alpha0 = muY.view(1)
        alpha = torch.cat((alpha0.view(1), alpha)).view(-1, 1)

        if return_all:
            return alpha, (alpha_all, sigma2_all, tau2_all, beta2_all)
        else:
            return alpha, None

    def _sample_alpha(self):
        if self.estimator == 'bayes':
            sigma2 = 1. / np.random.gamma(self.a_post, 1. / self.b_post)
            alpha = MultivariateNormal(loc=self.alpha_mean.view(-1), covariance_matrix=sigma2 * self.alpha_cov).sample()
            alpha = alpha.view(-1, 1).to(self.device, self.dtype)
        elif self.estimator == 'mle':
            if self.alpha_mle is None:
                raise Exception('You must firstly call the fit(X, y) function to be able to sample alpha')
            alpha = self.alpha_mle.to(self.device, self.dtype)
        elif self.estimator == 'horseshoe':
            if self.alpha_hs is None:
                raise Exception('You must firstly call the fit(X, y) function to be able to sample alpha')
            alpha = self.alpha_hs.to(self.device, self.dtype)
        elif self.estimator == 'sparse_horseshoe':
            if self.alpha_hs is None:
                raise Exception('You must firstly call the fit(X, y) function to be able to sample alpha')
            alpha = self.alpha_hs.view(-1)
            alpha[torch.abs(alpha) < self.sparse_horseshoe_threshold] = 0.
            alpha = alpha.view(-1, 1).to(self.device, self.dtype)
        else:
            raise NotImplementedError('Estimator is not implemented!')

        return alpha

    @staticmethod
    def _order_effects(onehot_x: torch.FloatTensor, order: int):
        # order_effects: Function computes data matrix for all coupling
        # orders to be added to linear regression model.

        # Find number of variables
        n_samp, n_vars = onehot_x.shape

        # Generate matrix to store results
        x_allpairs = onehot_x.clone()

        for ord_i in range(2, order + 1):

            # generate all combinations of indices (without diagonals)
            offdProd = np.array(list(combinations(np.arange(n_vars), ord_i)))

            # generate products of input variables
            x_comb = torch.zeros((n_samp, offdProd.shape[0], ord_i)).to(onehot_x)
            for j in range(ord_i):
                x_comb[:, :, j] = onehot_x[:, offdProd[:, j]]
            x_allpairs = torch.cat((x_allpairs, torch.prod(x_comb, axis=2)), axis=1)

        return x_allpairs


def standardise(X, y):
    # Standardize the covariates to have zero mean and x_i'x_i = 1

    # set params
    n = X.shape[0]
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0) * np.sqrt(n)

    # Standardize X's
    # sub_vector = np.vectorize(np.subtract)
    # X = sub_vector(X, meanX)
    # div_vector = np.vectorize(np.divide)
    # X = div_vector(X, stdX)

    # Standardize y's
    meany = y.mean()
    y = y - meany

    return X, meanX, stdX, y, meany


def fastmvg(Phi, alpha, D):
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference:
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778

    n, p = Phi.shape

    d = D.diag().to(Phi)
    u = torch.randn(p).to(Phi) * d.sqrt()
    delta = torch.randn(n).to(Phi)
    v = (Phi @ u + delta).view(-1, 1)
    Dpt = Phi.T * d[:, None]
    w = torch.linalg.solve(Phi @ Dpt + torch.eye(n).to(Phi), alpha - v)
    x = u + (Dpt @ w).view(-1)

    return x


def fastmvg_rue(X, XtX, y, Sigma_star):
    # Another sampler for multivariate Gaussians (small p) of the form
    #  N(mu, S), where
    #  mu = S Phi' y
    #  S  = inv(Phi'Phi + inv(D))
    #
    # Here, PtP = Phi'*Phi (X'X is precomputed)
    #

    p = X.shape[1]
    Sigma_star_inv = (1. / Sigma_star.diag()).diag()

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = torch.linalg.cholesky(XtX + Sigma_star_inv)
    except:
        mat = XtX + Sigma_star_inv
        Smat = (mat + mat.T) / 2.
        maxEig_Smat = torch.max(torch.linalg.eigvals(Smat).real)
        L = torch.linalg.cholesky(Smat + maxEig_Smat * 1e-15 * torch.eye(Smat.shape[0]).to(X))

    v = torch.linalg.solve(L, X.T @ y)
    m = torch.linalg.solve(L.T, v)
    w = torch.linalg.solve(L.T, torch.randn((p, 1)).to(X))

    x = m + w

    return x.view(-1)
