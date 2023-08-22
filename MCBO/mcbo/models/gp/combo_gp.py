# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from __future__ import annotations

import copy
import sys
import time
from datetime import timedelta
from typing import List, Optional

import numpy as np
import torch
from gpytorch.means import ConstantMean

from mcbo.models import ModelBase, EnsembleModelBase
from mcbo.models.gp.kernels import DiffusionKernel
from mcbo.search_space import SearchSpace
from mcbo.utils.sampling import univariate_slice_sampling
from mcbo.utils.training_utils import remove_repeating_samples
from mcbo.utils.training_utils import subsample_training_data


class ComboGPModel(ModelBase):
    """
    GP model with the Diffiusion kernel and Horseshoe prior on the ARD lengthscales. This model uses sampling to sample
    from the parameter's posterior.

    """

    @property
    def name(self) -> str:
        return "GP (Diffusion)"

    supports_cuda = True
    support_grad = True

    def __init__(self,
                 search_space: SearchSpace,
                 fourier_freq_list: List[torch.FloatTensor],
                 fourier_basis_list: List[torch.FloatTensor],
                 n_vertices: np.array,
                 adjacency_mat_list: List[torch.FloatTensor],
                 noise_lb: float = 1e-6,
                 n_burn: int = 0,
                 n_burn_init: int = 100,
                 max_training_dataset_size: int = 1000,
                 verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):

        super(ComboGPModel, self).__init__(search_space, 1, dtype, device)
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'The COMBO GP was only implemented for nominal and ordinal variables'

        self.noise_lb = noise_lb
        self.n_burn_init = n_burn_init
        self.np_dtype = 'float32' if self.dtype == torch.float64 else 'float64'
        self.verbose = verbose
        self.n_vertices = n_vertices
        self.adjacency_mat_list = adjacency_mat_list
        self.max_training_dataset_size = max_training_dataset_size
        self.n_burn = n_burn
        self.PROGRESS_BAR_LEN = 50

        # For numerical stability in exponential
        self.LOG_LOWER_BND = -12.0
        self.LOG_UPPER_BND = 20.0
        self.STABLE_MEAN_RNG = 1.0  # For sampling stability
        self.GRAPH_SIZE_LIMIT = 1024 + 2  # Hyperparameter for graph factorization

        # Initialise the algorithm
        self.is_initialised = False

        # Define the mean, likelihood and the kernel for the GP model
        self.mean = ConstantMean().to(dtype=self.dtype, device=self.device)
        self.log_likelihood_noise = torch.tensor([0.], dtype=self.dtype, device=self.device)

        fourier_freq_list = [fourier_freq.to(self.device, self.dtype) for fourier_freq in fourier_freq_list]
        fourier_basis_list = [fourier_basis.to(self.device, self.dtype) for fourier_basis in fourier_basis_list]
        self.kernel = DiffusionKernel(fourier_freq_list, fourier_basis_list).to(self.device, self.dtype)

        # Slice sampling is used to sample from the posterior distribution so there is no need to calculate gradients
        self.mean.constant.requires_grad = False

        # Variables required for inference
        self.output_min = None
        self.output_max = None
        self.mean_vec = None
        self.gram_mat = None
        self.cholesky = None
        self.jitter = 0

    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        # Normalise target values
        fit_y = (y - self.y_mean.to(y)) / self.y_std.to(y)

        # Add a small amount of noise to prevent training instabilities
        fit_y += 1e-6 * torch.randn_like(fit_y)
        return fit_y

    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        return fit_y * self.y_std.to(fit_y) + self.y_mean.to(fit_y)


    def fit(self, x: torch.FloatTensor, y: torch.FloatTensor, **kwargs):

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == self.num_out

        # Remove repeating data points
        x, y = remove_repeating_samples(x, y)

        # Determine if the dataset is not too large
        if len(y) > self.max_training_dataset_size:
            x, y = subsample_training_data(x, y, self.max_training_dataset_size)

        self.x = x.to(dtype=self.dtype, device=self.device)
        self._y = y.to(dtype=self.dtype, device=self.device)

        # Normalise target values
        self.fit_y = self.y_to_fit_y(y=self._y)

        self.output_min = torch.min(self.fit_y)
        self.output_max = torch.max(self.fit_y)

        if not self.is_initialised:
            self._initialise(self.fit_y)
            self.is_initialised = True

        if self.verbose:
            print('Sampling from the posterior...')
        self.sample_from_posterior(n_burn=self.n_burn, n_thin=1, verbose=self.verbose)

    def predict(self, x: torch.FloatTensor, **kwargs) -> (torch.FloatTensor, torch.FloatTensor):
        x = x.to(self.device, self.dtype)

        k_pred_train = self.kernel(x, self.x).evaluate()
        k_pred = self.kernel(x, diag=True)

        # cholesky is lower triangular matrix
        chol_solver = \
            torch.triangular_solve(torch.cat([k_pred_train.t(), self.mean_vec], 1), self.cholesky, upper=False)[0]
        chol_solve_k = chol_solver[:, :-1]
        chol_solve_y = chol_solver[:, -1:]

        pred_mean = torch.mm(chol_solve_k.t(), chol_solve_y) + self.mean(x).view(-1, 1)
        pred_quad = (chol_solve_k ** 2).sum(0).view(-1, 1)
        pred_var = k_pred - pred_quad
        pred_var = pred_var.clamp(min=1e-8)

        mu = self.fit_y_to_y(pred_mean).view(-1, self.num_out)
        var = (pred_var * self.y_std ** 2).view(-1, self.num_out)

        return mu, var

    def noise(self) -> torch.Tensor:
        return (self.log_likelihood_noise.exp() * self.y_std ** 2).view(self.num_out).detach()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> ComboEnsembleGPModel:
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        Args:
            device: target device
            dtype: target dtype

        Returns:
            self
        """
        if device is not None:
            assert isinstance(device, torch.device)

        self.device = device
        self.mean.to(device)
        self.kernel.log_amp = self.kernel.log_amp.to(device)
        self.kernel.log_beta = self.kernel.log_beta.to(device)
        self.kernel.fourier_freq_list = [freq.to(device, self.dtype) for freq in self.kernel.fourier_freq_list]
        self.kernel.fourier_basis_list = [basis.to(device, self.dtype) for basis in self.kernel.fourier_basis_list]
        return self

    def _initialise(self, y: torch.FloatTensor):
        """
        Function used to initialise the model the very first time it is fitted to data
        :param y:
        :return:
        """
        # Heuristic based model parameter initialisation
        output_mean = torch.mean(y).item()
        output_log_var = (0.5 * torch.var(y)).log().item()

        self.kernel.log_amp.fill_(output_log_var)
        self.mean.constant.fill_(output_mean)
        self.log_likelihood_noise = torch.tensor([output_mean / 1000.0])

        if self.verbose:
            print('Initialising GP model...')
        # Sampling begins with a burn in phase. No need to store the result as all model parameters are changed internally
        self.sample_from_posterior(n_burn=self.n_burn_init - 1, n_thin=1, verbose=self.verbose)

    def sample_from_posterior(self, n_burn: int = 0, n_thin: int = 1, verbose=False):
        """
        Function for sampling from the posterior distribution of the parameters. Note that this function changes the
        model parameters to the sampled values

        :param n_burn:
        :param n_thin:
        :param verbose:
        :return:
        """
        hyper_samples = []
        log_beta_samples = []

        log_beta_sample = self.kernel.log_beta.clone()

        n_sample_total = n_thin + n_burn
        n_digit = int(np.ceil(np.log(n_sample_total) / np.log(10)))
        start_time = time.time()
        for s in range(0, n_sample_total):
            log_amp, mean, log_likelihood_noise = self.slice_hyper()

            shuffled_beta_ind = list(range(len(self.adjacency_mat_list)))
            np.random.shuffle(shuffled_beta_ind)
            for beta_ind in shuffled_beta_ind:
                # In each sampler, model.kernel fourier_freq_list, fourier_basis_list are updated.
                log_beta_sample = self.slice_log_beta(log_beta=log_beta_sample, ind=beta_ind)

            if s >= n_burn and (s - n_burn + 1) % n_thin == 0:
                hyper_samples.append(torch.cat((log_amp, mean, log_likelihood_noise)))
                log_beta_samples.append(log_beta_sample.clone())

            if verbose:
                progress_mark_len = int((s + 1.0) / n_sample_total * self.PROGRESS_BAR_LEN)
                fmt_str = '(%s)   %3d%% (%' + str(n_digit) + 'd of %d) |' \
                          + '#' * progress_mark_len + '-' * (self.PROGRESS_BAR_LEN - progress_mark_len) + '|'
                progress_str = fmt_str % (str(timedelta(seconds=time.time() - start_time)).split(".")[0],
                                          int((s + 1.0) / n_sample_total * 100), s + 1, n_sample_total)
                sys.stdout.write(('\b' * len(progress_str)) + progress_str)
                sys.stdout.flush()
        if verbose:
            print('\n')

    def slice_hyper(self) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        """
        Slice sampling of kernel amplitude, constant mean and likelihood noise
        :return:
        """
        # Randomly shuffling order can be considered, here the order is in const_mean, kernel_amp, noise_var
        mean = self.slice_const_mean()
        log_amp = self.slice_kernelamp()
        log_likelihood_noise = self.slice_noisevar()

        return log_amp, mean, log_likelihood_noise

    def slice_const_mean(self) -> torch.FloatTensor:
        """
        Slice sampling for constant mean

        :return:
        """
        output_min = self.output_min.item()
        output_max = self.output_max.item()
        log_amp = self.kernel.log_amp.item()
        log_likelihood_noise = self.log_likelihood_noise.item()
        log_beta = self.kernel.log_beta.clone()

        def logp(constmean):
            """
            :param constmean: numeric(float)
            :return: numeric(float)
            """
            log_prior = self.log_prior_const_mean(constmean, output_min=output_min, output_max=output_max)
            if np.isinf(log_prior):
                return log_prior
            log_likelihood = -self.negative_log_likelihood(constmean, log_amp, log_beta, log_likelihood_noise).item()
            return log_prior + log_likelihood

        constmean_init = self.mean.constant.item()
        constmean = univariate_slice_sampling(logp, constmean_init, dtype=self.np_dtype)
        self.mean.constant.fill_(constmean)

        return torch.tensor([constmean], dtype=self.dtype, device=self.device)

    def slice_noisevar(self) -> torch.FloatTensor:
        """
        Slice sampling for log noise variance, this function

        """

        log_amp = self.kernel.log_amp.item()
        log_beta = self.kernel.log_beta.clone()
        constmean = self.mean.constant.item()

        def logp(log_noise_var):
            """
            :param log_noise_var: numeric(float)
            :return: numeric(float)
            """
            log_prior = self._log_prior_noisevar(log_noise_var)
            if np.isinf(log_prior):
                return log_prior

            log_likelihood = -self.negative_log_likelihood(constmean, log_amp, log_beta, log_noise_var).item()
            return log_prior + log_likelihood

        log_likelihood_noise_init = self.log_likelihood_noise.item()
        log_likelihood_noise = univariate_slice_sampling(logp, log_likelihood_noise_init)
        log_likelihood_noise = np.maximum(log_likelihood_noise, np.log(self.noise_lb))
        log_likelihood_noise = torch.tensor([log_likelihood_noise], dtype=self.dtype, device=self.device)
        self.log_likelihood_noise = log_likelihood_noise

        return log_likelihood_noise

    def slice_kernelamp(self) -> torch.FloatTensor:
        """
        Slice sampling for kernel amplitude
        :return:
        """

        output_var = self.fit_y.var().item()
        kernel_min = np.prod(
            [torch.mean(torch.exp(-fourier_freq[-1])).item() / torch.mean(torch.exp(-fourier_freq)).item() for
             fourier_freq in self.kernel.fourier_freq_list])
        kernel_max = np.prod(
            [torch.mean(torch.exp(-fourier_freq[0])).item() / torch.mean(torch.exp(-fourier_freq)).item() for
             fourier_freq in self.kernel.fourier_freq_list])

        log_beta = self.kernel.log_beta.clone()
        log_likelihood_noise = self.log_likelihood_noise.item()
        constmean = self.mean.constant.item()

        def logp(log_amp):
            """
            :param log_amp: numeric(float)
            :return: numeric(float)
            """
            log_prior = self.log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max)
            if np.isinf(log_prior):
                return log_prior
            log_likelihood = -self.negative_log_likelihood(constmean, log_amp, log_beta, log_likelihood_noise).item()
            return log_prior + log_likelihood

        log_amp_init = self.kernel.log_amp.item()
        log_amp = univariate_slice_sampling(logp, log_amp_init, dtype=self.np_dtype)
        log_amp = torch.tensor([log_amp], dtype=self.dtype, device=self.device)
        self.kernel.log_amp = log_amp
        return log_amp

    def slice_log_beta(self, log_beta: torch.FloatTensor, ind: int):
        """
        Slice sampling for log_beta[ind]
        """

        log_amp = self.kernel.log_amp.item()
        log_likelihood_noise = self.log_likelihood_noise.item()
        constmean = self.mean.constant.item()

        def logp(log_beta_i):
            """
            Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
            :param log_beta_i: numeric(float)
            :return: numeric(float)
            """
            log_prior = self.log_prior_log_beta(log_beta_i)
            if np.isinf(log_prior):
                return log_prior
            self.kernel.log_beta[ind] = log_beta_i
            log_likelihood = -self.negative_log_likelihood(constmean, log_amp, log_beta, log_likelihood_noise).item()
            return log_prior + log_likelihood

        beta_i_init = log_beta[ind].item()
        beta_i = univariate_slice_sampling(logp, beta_i_init, dtype=self.np_dtype)
        log_beta[ind] = torch.tensor(beta_i)
        return log_beta

    def log_prior_const_mean(self, const_mean: float, output_min: float, output_max: float):
        """
        Calculation of log prior of the constant mean
        :param const_mean:
        :param output_min:
        :param output_max:
        :return:
        """
        output_mid = (output_min + output_max) / 2.0
        output_rad = (output_max - output_min) * self.STABLE_MEAN_RNG / 2.0
        # Unstable parameter in sampling
        if const_mean < output_mid - output_rad or output_mid + output_rad < const_mean:
            return -float('inf')
        # Truncated Gaussian
        stable_dev = output_rad / 2.0 + 1e-7
        return -np.log(stable_dev) - 0.5 * (const_mean - output_mid) ** 2 / stable_dev ** 2

    def _log_prior_noisevar(self, log_noise_var: float):
        """
        Calculation of log prior of the noise variance
        :param log_noise_var:
        :return:
        """
        if log_noise_var < self.LOG_LOWER_BND or min(self.LOG_UPPER_BND, np.log(10000.0)) < log_noise_var:
            return -float('inf')
        return np.log(np.log(1.0 + (0.1 / np.exp(log_noise_var)) ** 2))

    def log_prior_kernelamp(self, log_amp: float, output_var: float, kernel_min: float, kernel_max: float):
        """
        Calculation of log prior of log kernel amplitude
        :param log_amp:
        :param output_var: numeric(float)
        :param kernel_min: numeric(float)
        :param kernel_max: numeric(float)
        :return:
        """
        if log_amp < self.LOG_LOWER_BND or min(self.LOG_UPPER_BND, np.log(10000.0)) < log_amp:
            return -float('inf')
        log_amp_lower = np.log(output_var) - np.log(kernel_max)
        log_amp_upper = np.log(output_var) - np.log(max(kernel_min, 1e-100))
        log_amp_mid = 0.5 * (log_amp_upper + log_amp_lower)
        log_amp_rad = 0.5 * (log_amp_upper - log_amp_lower)
        log_amp_std = log_amp_rad / 2.0
        return -np.log(log_amp_std) - 0.5 * (log_amp - log_amp_mid) ** 2 / log_amp_std ** 2

    def log_prior_log_beta(self, log_beta_i):
        """
        Calculation of log prior of log beta

        :param log_beta_i: numeric(float), ind-th element of 'log_beta'
        :return:
        """
        if log_beta_i < self.LOG_LOWER_BND or min(self.LOG_UPPER_BND, np.log(100.0)) < log_beta_i:
            return -float('inf')

        ## Horseshoe prior
        tau = 5.0
        return np.log(np.log(1.0 + 2.0 / (np.exp(log_beta_i) / tau) ** 2))

    def gram_mat_update(self, mean: float, log_amp: float, log_beta: torch.FloatTensor, log_likelihood_noise: float):
        """
        Function used to update the gram matrix based on new model parameters. This function will use the stored self.x
        and self.fit_y.

        :param mean:
        :param log_amp:
        :param log_beta:
        :param log_likelihood_noise:
        :return:
        """
        # Note that this function updates the model parameters
        self.kernel.log_amp = torch.tensor([log_amp], dtype=self.dtype, device=self.device)
        self.kernel.log_beta = log_beta.to(self.device, self.dtype)
        self.mean.constant.fill_(mean)
        self.log_likelihood_noise = torch.tensor([log_likelihood_noise], dtype=self.dtype, device=self.device)

        self.mean_vec = self.fit_y.view(-1, 1) - self.mean(self.x).view(-1, 1)
        self.gram_mat = self.kernel(self.x).evaluate() + torch.exp(self.log_likelihood_noise).repeat(
            self.x.size(0)).diag()

    def cholesky_update(self, constmean: float, log_amp: float, log_beta: torch.FloatTensor,
                        log_likelihood_noise: float):
        """
        Function used to update the Cholesky matrix
        :param constmean
        :param log_amp:
        :param log_beta:
        :param log_likelihood_noise:
        :return:
        """
        self.gram_mat_update(constmean, log_amp, log_beta, log_likelihood_noise)

        eye_mat = torch.diag(self.gram_mat.new_ones(self.gram_mat.size(0)))
        for jitter_const in [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            chol_jitter = torch.trace(self.gram_mat).item() * jitter_const
            try:
                # cholesky is lower triangular matrix
                self.cholesky = torch.linalg.cholesky(self.gram_mat + eye_mat * chol_jitter)
                self.jitter = chol_jitter
                return
            except RuntimeError:
                pass
        raise RuntimeError('Absolute entry values of Gram matrix are between %.4E~%.4E with trace %.4E' %
                           (torch.min(torch.abs(self.gram_mat)).item(), torch.max(torch.abs(self.gram_mat)).item(),
                            torch.trace(self.gram_mat).item()))

    def negative_log_likelihood(self, constmean: float, log_amp: float, log_beta: torch.FloatTensor,
                                log_likelihood_noise: float) -> torch.FloatTensor:
        """
        Function used to calculate the negative log likelihood
        :param constmean:
        :param log_amp:
        :param log_beta:
        :param log_likelihood_noise:
        :return:
        """
        self.cholesky_update(constmean, log_amp, log_beta, log_likelihood_noise)

        # cholesky is lower triangular matrix
        mean_vec_sol = torch.triangular_solve(self.mean_vec, self.cholesky, upper=False)[0]
        nll = 0.5 * torch.sum(mean_vec_sol ** 2) + torch.sum(
            torch.log(torch.diag(self.cholesky))) + 0.5 * self.fit_y.size(0) * np.log(2 * np.pi)

        return nll


class ComboEnsembleGPModel(EnsembleModelBase):

    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        return fit_y

    @property
    def name(self) -> str:
        return "GP (Diffusion)"

    def __init__(self,
                 search_space: SearchSpace,
                 fourier_freq_list: List[torch.FloatTensor],
                 fourier_basis_list: List[torch.FloatTensor],
                 n_vertices: np.array,
                 adjacency_mat_list: List[torch.FloatTensor],
                 n_models: int = 10,
                 noise_lb: float = 1e-6,
                 n_burn: int = 0,
                 n_burn_init: int = 100,
                 max_training_dataset_size: int = 1000,
                 verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):

        self.combo_gp = ComboGPModel(search_space, fourier_freq_list, fourier_basis_list, n_vertices,
                                     adjacency_mat_list, noise_lb, n_burn, n_burn_init,
                                     max_training_dataset_size, verbose, dtype, device)

        super(ComboEnsembleGPModel, self).__init__(search_space, 1, n_models, dtype, device)

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        self.models = []
        for _ in range(self.num_models):
            self.combo_gp.fit(x, self.y_to_fit_y(y))
            self.models.append(copy.deepcopy(self.combo_gp))

        return

    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):

        mu = torch.zeros((len(x), self.num_out, self.num_models)).to(x)
        var = torch.zeros((len(x), self.num_out, self.num_models)).to(x)

        for i, model in enumerate(self.models):
            mu[..., i], var[..., i] = model.predict(x)

        return mu.mean(-1), var.sum(-1) / len(self.models) ** 2
