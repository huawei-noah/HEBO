# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Union, List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from gpytorch import settings, lazify, delazify
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel, MaternKernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from torch.nn import ModuleList

from mcbo.search_space import SearchSpace
from mcbo.utils.hed_utils import diverse_random_dict_sample


class MixtureKernel(Kernel):

    @property
    def name(self) -> str:
        numeric_kernel_name = get_numeric_kernel_name(self.numeric_kernel)
        nominal_kernel_name = get_nominal_kernel_name(self.categorical_kernel)
        name = f"{numeric_kernel_name} and {nominal_kernel_name}"
        return name

    has_lengthscale = True

    def __init__(self, search_space: SearchSpace, numeric_kernel: Kernel, categorical_kernel: Kernel,
                 lamda: float = 0.5, **kwargs):

        super(MixtureKernel, self).__init__(**kwargs)
        if search_space is None:
            self.num_dims = kwargs['num_dims']
            self.nominal_dims = kwargs['nominal_dims']
        else:
            self.num_dims = search_space.cont_dims + search_space.disc_dims
            self.nominal_dims = search_space.nominal_dims

        self.optimize_lamda = lamda is None
        self.fixed_lamda = lamda if not self.optimize_lamda else None

        # Register parameter lambda and its constraint
        self.register_parameter(name='raw_lamda', parameter=torch.nn.Parameter(torch.ones(1)))
        self.register_constraint('raw_lamda', Interval(0., 1.))

        # Initialise the
        self.categorical_kernel = categorical_kernel
        self.numeric_kernel = numeric_kernel

        self.has_lengthscale = self.categorical_kernel.has_lengthscale or self.numeric_kernel.has_lengthscale

    @property
    def lamda(self):
        if self.optimize_lamda:
            return self.raw_lamda_constraint.transform(self.raw_lamda)
        else:
            return self.fixed_lamda

    @lamda.setter
    def lamda(self, value):
        self._set_lamda(value)

    def _set_lamda(self, value):
        if self.optimize_lamda:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value).to(self.raw_lamda)
            self.initialize(raw_lamda=self.raw_lamda_constraint.inverse_transform(value))
        else:
            # Manually restrict the value of lamda between 0 and 1.
            if value <= 0:
                self.fixed_lamda = 0.
            elif value >= 1:
                self.fixed_lamda = 1.
            else:
                self.fixed_lamda = value

    def forward(self, x1, x2, diag=False, **params):

        assert x1.shape[1] == len(self.num_dims) + len(self.nominal_dims)

        k_cat = self.categorical_kernel(x1, x2, diag, **params)
        k_cont = self.numeric_kernel(x1, x2, diag, **params)

        return (1. - self.lamda) * (k_cat + k_cont) + self.lamda * k_cat * k_cont


class Overlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    @property
    def name(self) -> str:
        return "O"

    def __init__(self, **kwargs):
        super(Overlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).to(x1)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).to(x1)
        return k_cat.to(x1)


class TransformedOverlap(Overlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    @property
    def name(self) -> str:
        return "TO"

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        diff = x1[:, None] - x2[None, :]
        diff[torch.abs(diff) > 1e-5] = 1
        diff1 = torch.logical_not(diff).to(x1)

        def rbf(d, ard):
            if ard:
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).to(x1)
        return k_cat.to(x1)


class OrdinalKernel(Kernel):
    """
    The ordinal version of TransformedCategorical2 kernel (replace the Kronecker delta with
    the distance metric).
    config: the number of vertices per dimension
    """

    def __init__(self, config, **kwargs):
        super(OrdinalKernel, self).__init__(has_lengthscale=True, **kwargs)
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config).view(-1)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # expected x1 and x2 are of shape N x D respectively
        diff = (x1[:, None] - x2[None, :]) / self.config
        dist = 1. - torch.abs(diff)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.exp(
                torch.sum(
                    dist * self.lengthscale, dim=-1
                ) / torch.sum(self.lengthscale)
            )
        else:
            k_cat = torch.exp(
                self.lengthscale * torch.sum(dist, dim=-1) / x1.shape[1]
            )
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class SubStringKernel(Kernel):
    """
    Code based on https://github.com/beckdaniel/flakes
    We make following changes
    1) provide kernel normalization to make meaningful comparisons between strings of different lengths
    2) changed structure and conventions to match our Tree kernel implementation
    3) simplified to only allow one-hot encoding of alphabet (i.e remove support for pre-trained embeddings)
    4) a collection of performance tweaks to improve vectorization
    """

    @property
    def name(self) -> str:
        return "SSK"

    def __init__(self, seq_length: int, alphabet_size: int, gap_decay=.5, match_decay=.8,
                 max_subsequence_length: int = 3, normalize=False, **kwargs):
        super(SubStringKernel, self).__init__(has_lengthscale=False, **kwargs)

        self.register_parameter(name='match_decay', parameter=torch.nn.Parameter(torch.tensor(match_decay)))
        self.register_parameter(name='gap_decay', parameter=torch.nn.Parameter(torch.tensor(gap_decay)))
        self.register_constraint("gap_decay", Interval(0, 1))
        self.register_constraint("match_decay", Interval(0, 1))
        self.max_subsequence_length = max_subsequence_length

        # store additional kernel parameters
        self.maxlen = seq_length
        self.alphabet_size = alphabet_size
        self.normalize = normalize

        self.tril = torch.triu(torch.ones((self.maxlen, self.maxlen), dtype=torch.double), diagonal=1)
        self.exp = torch.ones(self.maxlen, self.maxlen, dtype=torch.int)
        for i in range(self.maxlen - 1):
            self.exp[i, i + 1:] = torch.arange(self.maxlen - i - 1)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.tril.to(device=device, dtype=dtype)
        self.exp.to(device=device)
        return super().to(device=device, dtype=dtype)

    def K_diag(self, x: torch.Tensor):
        r"""
        The diagonal elements of the string kernel are always unity (due to normalisation)
        """
        return torch.ones(x.shape[:-1], dtype=self.dtype)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, last_dim_is_batch=False, **params):
        r"""
        Vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

        # Turn our inputs into lists of integers using one-hot embedding
        # first split up strings and pad to fixed length and prep for gpu
        # pad until all have length of self.maxlen
        if x2 is None or self.training:
            x2 = x1
            symmetric = True
        else:
            symmetric = False

        # keep track of original input sizes
        x1_shape = x1.shape[0]
        x2_shape = x2.shape[0]

        # prep the decay tensor D
        D = self._precalc().to(x1)

        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        x1 = torch.nn.functional.one_hot(x1.to(int), self.alphabet_size).to(x1)
        x2 = torch.nn.functional.one_hot(x2.to(int), self.alphabet_size).to(x2)

        # get indices of all possible pairings from X1 and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indices_2, indices_1 = torch.meshgrid(torch.arange(0, x2.shape[0]), torch.arange(0, x1.shape[0]), indexing='ij')
        indices = torch.cat([torch.reshape(indices_1.T, (-1, 1)), torch.reshape(indices_2.T, (-1, 1))], axis=1)

        # if symmetric then only calc upper matrix (fill in rest later)
        if symmetric:
            indices = indices[indices[:, 1] >= indices[:, 0]]

        x1_full = torch.repeat_interleave(x1.unsqueeze(0), len(indices), dim=0)[np.arange(len(indices)), indices[:, 0]]
        x2_full = torch.repeat_interleave(x2.unsqueeze(0), len(indices), dim=0)[np.arange(len(indices)), indices[:, 1]]

        if not symmetric:
            # also need to calculate some extra kernel evals for the normalization terms
            x1_full = torch.cat([x1_full, x1, x2], 0)
            x2_full = torch.cat([x2_full, x1, x2], 0)

        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        s = torch.matmul(x1_full, torch.transpose(x2_full, 1, 2))

        # store squared match coef
        match_sq = self.match_decay ** 2

        kp = torch.ones(*[s.shape[0], self.maxlen, self.maxlen]).to(s)

        # do all remaining steps
        for _ in torch.arange(self.max_subsequence_length - 1):
            kp = torch.multiply(s, kp)
            kp = match_sq * kp
            kp = torch.matmul(kp, D)
            kp = torch.matmul(D.T, kp)

        # final kernel calc
        kp = torch.multiply(s, kp)
        k = kp.sum((-2, -1)).unsqueeze(1) * match_sq

        # put results into the right places in the gram matrix and normalize
        if symmetric:
            # if symmetric then only put in top triangle (inc diag)
            mask = torch.triu(torch.ones((x1_shape, x2_shape)), 0).to(s)
            non_zero = mask > 0
            k_results = torch.zeros((x1_shape, x2_shape)).to(s)
            k_results[non_zero] = k.squeeze()
            # add in mising elements (lower diagonal)
            k_results = k_results + k_results.T - torch.diag(k_results.diag())

            if self.normalize:
                # normalise
                x_diag_ks = torch.diag(k_results)
                norm = torch.matmul(x_diag_ks[:, None], x_diag_ks[None, :])
                k_results = torch.divide(k_results, torch.sqrt(norm))
        else:
            # otherwise can just reshape into gram matrix
            # but first take extra kernel calcs off end of k

            # COULD SPEED THIS UP FOR PREDICTIONS, AS MANY NORM TERMS ALREADY IN GRAM

            x_diag_ks = k[x1_shape * x2_shape:x1_shape * x2_shape + x1_shape].flatten()

            x2_diag_ks = k[-x2_shape:].flatten()

            k = k[0:x1_shape * x2_shape]
            k_results = k.reshape(x1_shape, x2_shape)

            if self.normalize:
                # normalise
                norm = torch.matmul(x_diag_ks[:, None], x2_diag_ks[None, :])
                k_results = torch.divide(k_results, torch.sqrt(norm))

        if diag:
            return k_results.diag()

        return k_results

    def _precalc(self):
        r"""
        Precalc D matrix as required for kernel calcs
        following notation from Beck (2017)
        """
        return torch.pow(self.gap_decay.to(self.tril) * self.tril, self.exp.to(device=self.tril.device))


class DiffusionKernel(Kernel):
    """
    Usually Graph Kernel means a kernel between graphs, here this kernel is a kernel between vertices on a graph
    Edge scales are not included in the module, instead edge weights of each subgraphs is used to calculate frequencies (fourier_freq)
    """

    @property
    def name(self) -> str:
        return "Diffusion"

    def __init__(self, fourier_freq_list, fourier_basis_list, **kwargs):
        super(DiffusionKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.log_amp = torch.zeros(1).to(fourier_freq_list[0])
        self.log_beta = torch.zeros(len(fourier_freq_list)).to(fourier_freq_list[0])
        self.fourier_freq_list = fourier_freq_list
        self.fourier_basis_list = fourier_basis_list

    def forward(self, X1: torch.Tensor, X2: torch.Tensor, diag=False, last_dim_is_batch=False, **params):
        """
        Args:
            X1: each row is a vector with vertex numbers starting from 0 for each
            X2: each row is a vector with vertex numbers starting from 0 for each

        """

        stabilizer = 0
        if (X1.shape == X2.shape) and (X1 == X2).all():
            X2 = X1
            if diag:
                stabilizer = 1e-6 * X1.new_ones(X1.size(0), 1, dtype=X1.dtype)
            else:
                stabilizer = torch.diag(1e-6 * X1.new_ones(X1.size(0), dtype=X1.dtype))

        full_gram = 1
        for i in range(len(self.fourier_freq_list)):
            beta = torch.exp(self.log_beta[i].to(dtype=X1.dtype, device=X1.device))
            fourier_freq = self.fourier_freq_list[i].to(dtype=X1.dtype, device=X1.device)
            fourier_basis = self.fourier_basis_list[i].to(dtype=X1.dtype, device=X1.device)

            subvec1 = fourier_basis[X1[:, i].long()]
            subvec2 = fourier_basis[X2[:, i].long()]
            freq_transform = torch.exp(-beta * fourier_freq)

            if diag:
                factor_gram = torch.sum(subvec1 * freq_transform.unsqueeze(0) * subvec2, dim=1, keepdim=True)
            else:
                factor_gram = torch.matmul(subvec1 * freq_transform.unsqueeze(0), subvec2.t())

            # HACK for numerical stability for scalability
            full_gram *= factor_gram / torch.mean(freq_transform)

        res = torch.exp(self.log_amp) * (full_gram + stabilizer)

        return res


class ConditionalTransformedOverlapKernel(Kernel):
    r"""
     - if not ARD: $k(x, x') =$
     $$ \exp(\frac{\lambda}{n} \sum_{i=1}^n \mathbb{1} [x_i = x'_i] K^{H_{x_i}}(x^{H_{x_i}}, {x'}^{H_{x_i}})) $$
       - if ARD: $k(x, x') =$
     $$ \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i \mathbb{1} [x_i = x'_i] K^{H_{x_i}}(x^{H_{x_i}}, {x'}^{H_{x_i}})) $$

    Example:
        >>> hyp_kernels = [
        >>>    MaternKernel(active_dims=torch.tensor([0, 1]), ard_num_dims=2),
        >>>    MaternKernel(active_dims=torch.tensor([2, 3, 4]), ard_num_dims=3)
        >>>]
        >>> seq_indices = np.arange(3)
        >>> param_indices = np.arange(3, 8)
        >>> n_categories = 4
        >>> map_cat_to_kernel_ind = torch.tensor([-1, 0, 1, -1])
        >>> conditional_transf_cat_kern = ConditionalTransformedOverlapKernel(
        >>>     *hyp_kernels,
        >>>     seq_indices=seq_indices,
        >>>     param_indices=param_indices,
        >>>     n_categories=n_categories,
        >>>     map_cat_to_kernel_ind=map_cat_to_kernel_ind,
        >>>     ard_num_dims=len(seq_indices)
        >>> )
        >>> x1 = torch.zeros(20, 8)
        >>> x1[:, seq_indices] = torch.randint(0, n_categories, (len(x1), len(seq_indices))).to(x1)
        >>> x1[:, param_indices] = torch.rand(len(x1), len(param_indices))
        >>> conditional_transf_cat_kern_x1_x1 = conditional_transf_cat_kern(x1)
    """

    has_lengthscale = True

    @property
    def name(self) -> str:
        return "Cond-TO"

    def __init__(self, *hyp_kernels: Kernel, seq_indices: Union[np.ndarray, torch.Tensor],
                 param_indices: Union[np.ndarray, torch.Tensor], n_categories: int,
                 map_cat_to_kernel_ind: Union[List[int], np.ndarray, torch.Tensor],
                 ard_num_dims: Optional[int] = None,
                 ):
        """
        Args:
            hyp_kernels: list of kernels over the hyperparameters associated to each category
                         (their active dims should correspond to the indices of the relevant hyperparamters in the
                         vector of hyyperparameters extracted with param_indices)
            seq_indices: list of dimensions corresponding to the sequence, not to the hyperparameters
            param_indices: list of dimensions corresponding to the hyperparameters
            n_categories: total number of categories
            map_cat_to_kernel_ind: list containing at index `cat` the index `k_cat` such that hyperparams of category
                                   `cat` should be dealt with by kernel `hyp_kernels[k_cat]`. If a category is
                                   associated to no hyperparam, the corresponding index should be `-1`
        """
        if not isinstance(map_cat_to_kernel_ind, torch.Tensor):
            map_cat_to_kernel_ind = torch.tensor(map_cat_to_kernel_ind, dtype=torch.int)
        if len(map_cat_to_kernel_ind) == n_categories:
            map_cat_to_kernel_ind = torch.cat(
                [map_cat_to_kernel_ind, -2 * torch.ones(1).to(map_cat_to_kernel_ind)]).to(
                int)
        if ard_num_dims:
            assert ard_num_dims == len(seq_indices), (ard_num_dims, len(seq_indices), seq_indices)
        super(ConditionalTransformedOverlapKernel, self).__init__(has_lengthscale=True,
                                                                  ard_num_dims=ard_num_dims)
        self.kernels = ModuleList(hyp_kernels)
        self.n_cats = n_categories
        self.seq_indices = seq_indices
        self.param_indices = param_indices
        assert map_cat_to_kernel_ind[-1] == -2, f"Absence of match -> -2: {map_cat_to_kernel_ind}"
        assert torch.all(map_cat_to_kernel_ind < len(
            self.kernels)), f"Mapping to an unreachable kernel index: {map_cat_to_kernel_ind.max()} " \
                            f"in {len(hyp_kernels)} kernels"
        self.map_cat_to_kernel_ind = map_cat_to_kernel_ind

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', check_output: bool = False, **params):
        """
        Return K(x1, x2)
        """
        assert self.num_outputs_per_input(x1, x2) == 1, self.num_outputs_per_input(x1, x2)
        assert not last_dim_is_batch
        assert x1.ndim <= 2 and x2.ndim <= 2, (x1.shape, x2.shape)

        self.map_cat_to_kernel_ind = self.map_cat_to_kernel_ind.to(device=x1.device)
        diff = x1[:, self.seq_indices][:, None] - x2[:, self.seq_indices][None, :]
        diff = torch.abs(diff) > 1e-5
        diff = torch.logical_not(diff).to(x1) * 2 - 1  # -1 / 1 values (no match / match)
        diff = (1 + x1[:, self.seq_indices][:,
                    None]) * diff - 1  # replace `1` by the category, keep negative values negative
        diff[diff < 0] = -1  # set negative values to -1
        diff = self.map_cat_to_kernel_ind[
            diff.to(int)]  # non-negative values: get the kernel index, -2: no match, -1: no hyp

        assert diff.shape == (x1.shape[0], x2.shape[0], len(self.seq_indices))

        cond_diff_matrix = torch.zeros_like(diff).to(x1)

        for ind_k in range(len(self.kernels)):
            filtre = (diff == ind_k)
            if filtre.sum() > 0:
                cond_diff_matrix += filtre * \
                                    delazify(
                                        self.kernels[ind_k](x1[..., self.param_indices], x2[..., self.param_indices],
                                                            diag=False, **params))[..., None]

        cond_diff_matrix[diff == -2] = 0  # no match -> similarity is 0
        cond_diff_matrix[diff == -1] = 1  # match but no hyp -> perfect similarity

        if check_output:
            manual_out = self.manual_get_cond_diff_matrix(x1, x2, diag=diag, **params)
            assert torch.allclose(manual_out, cond_diff_matrix)
            print("Check of conditional distance matrix passed!")

        def rbf(d, ard):
            if ard:
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(cond_diff_matrix, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(cond_diff_matrix, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)

        if diag:
            return torch.diag(k_cat).to(x1)
        return k_cat.to(x1)

    def manual_get_cond_diff_matrix(self, x1, x2, diag=False, **params) -> torch.Tensor:
        """
        Compute K(x1, x2) without using vector tricks [used for debugging only]
        """
        cond_diff_matrix = torch.zeros((x1.shape[0], x2.shape[0], len(self.seq_indices))).to(x1)

        K: List[torch.Tensor] = [self.kernels[k_ind](x1[..., self.param_indices], x2[..., self.param_indices],
                                                     diag=False, **params) for k_ind in range(len(self.kernels))]

        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                for k in range(len(self.seq_indices)):
                    if x1[i, k] == x2[j, k]:
                        assert torch.round(x1[i, k]) == x1[i, k]
                        kern_ind = self.map_cat_to_kernel_ind[x1[i, k].to(int)]
                        if kern_ind == -1:
                            cond_diff_matrix[i, j, k] = 1
                        else:
                            cond_diff_matrix[i, j, k] = K[kern_ind][i, j]

        if diag:
            return torch.diag(cond_diff_matrix)
        return cond_diff_matrix

    def get_lengthcales_numerical_dims(self) -> torch.Tensor:
        weights = []
        for kernel in self.kernels:
            if isinstance(kernel, ScaleKernel):
                kernel = kernel.base_kernel
            if hasattr(kernel, "numeric_kernel"):
                kernel = kernel.numeric_kernel
                if isinstance(kernel, ScaleKernel):
                    kernel = kernel.base_kernel
            if isinstance(kernel, MaternKernel) or isinstance(kernel, RBFKernel):
                weights.append(kernel.lengthscale.flatten())
        weights = torch.cat(weights).flatten()
        if len(weights) > 0:
            weights = weights.unsqueeze(0)
        return weights

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        """ Same as `__call__` in `Kernel` except the check on the `ard_num_dims` """
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        # Do not check that ard_num_dims matches the supplied number of dimensions
        # if settings.debug.on():
        #     if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
        #         raise RuntimeError(
        #             "Expected the input to have {} dimensionality "
        #             "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
        #         )

        if diag:
            res = super(ConditionalTransformedOverlapKernel, self).__call__(x1_, x2_, diag=True,
                                                                            last_dim_is_batch=last_dim_is_batch,
                                                                            **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diag()
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = lazify(super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params))
            return res


class HEDKernel(Kernel):
    """
    Input warping with Hamming embedding via dictionaries compatible with any base_kernel on continuous variables.

    Note that the dictionary vectors are resampled each time train(True) method is called (triggers resampling at each
    GP fit).
    """
    has_lengthscale = False  # base kernel can have lengthscales, but there is no lengthscale onto the original space.

    @property
    def name(self) -> str:
        name = "HED"
        num_k_name = get_numeric_kernel_name(kernel=self.base_kernel)
        name += f"-{num_k_name}"
        return name

    def __init__(self, base_kernel: Kernel, hed_num_embedders: int, n_cats_per_dim: List[int],
                 active_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(active_dims=active_dims)
        self.hed_num_embedders = hed_num_embedders
        self.n_cats_per_dim = n_cats_per_dim
        self.base_kernel = base_kernel
        self.hed_embedders = self.sample_embedders()

    def kernel_name(self) -> str:
        return f"HED-{self.hed_num_embedders}D-{self.base_kernel.__class__.__name__}"

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        phi_x1 = self.embed(x=x1)
        phi_x2 = self.embed(x=x2)
        return self.base_kernel(x1=phi_x1, x2=phi_x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def sample_embedders(self) -> torch.Tensor:
        return torch.tensor(diverse_random_dict_sample(m=self.hed_num_embedders, n_cats_per_dim=self.n_cats_per_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed `x` using dictionary vectors A

        Args:
            x: tensor (batch_size, d) to embed

        Returns:
            embeddings: tensor phi(x), where phi(x)[i, j] = hamming_dist(x[i], A[j])
        """
        assert x.shape[-1] == len(self.n_cats_per_dim), (x.shape, len(self.n_cats_per_dim))

        to_flatten = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            to_flatten = True
        embeddings = torch.cdist(x1=x.float(), x2=self.hed_embedders.to(x).float(), p=0)

        target_shape = (x.shape[0], self.hed_num_embedders)
        assert embeddings.shape == target_shape, (embeddings.shape, target_shape)

        if to_flatten:
            embeddings = embeddings.squeeze()
        return embeddings / len(self.n_cats_per_dim)  # normalise

    def train(self, mode=True):
        if mode:
            # resample the hed_embedders
            self.hed_embedders = self.sample_embedders()
        self.base_kernel.train(mode=mode)


def get_numeric_kernel_name(kernel: Kernel) -> str:
    if isinstance(kernel, ScaleKernel):
        kernel = kernel.base_kernel
    if isinstance(kernel, MaternKernel):
        num_k_name = "mat"
        if kernel.nu == 0.5:
            num_k_name += "12"
        elif kernel.nu == 1.5:
            num_k_name += "32"
        elif kernel.nu == 2.5:
            num_k_name += "52"
        else:
            raise ValueError(kernel.nu)
    elif isinstance(kernel, RBFKernel):
        num_k_name = "rbf"
    elif isinstance(kernel, DecompositionKernel):
        num_k_name = kernel.base_kernel_num
    else:
        raise ValueError(kernel.__class__)
    return num_k_name


def get_nominal_kernel_name(kernel: Kernel) -> str:
    if isinstance(kernel, ScaleKernel):
        kernel = kernel.base_kernel
    return kernel.name


class DecompositionKernel(Kernel):
    r"""
        Fast kernel for decompositions
    """

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        return m._set_outputscale(v)

    @outputscale.setter
    def outputscale(self, value: Union[float, torch.tensor]):
        self._set_outputscale(value)

    def _set_outputscale(self, value: Union[float, torch.tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def __init__(
            self,
            decomposition: List[List[int]],
            base_kernel_num: str,
            base_kernel_kwargs_num: Optional[Dict[str, Any]],
            base_kernel_nom: str,
            base_kernel_kwargs_nom: Optional[Dict[str, Any]],
            search_space: SearchSpace,
            num_lengthscale_constraint: torch.nn.Module,
            nom_lengthscale_constraint: torch.nn.Module,
            outputscale_prior: Optional[Prior] = None,
            outputscale_constraint: Optional[Interval] = None,
            lengthscale_prior: Optional[Prior] = None,
            lengthscale_constraint: Optional[Interval] = None,
            hed: bool = False,
            hed_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.search_space = search_space
        kernel_dict = {}

        self.hed_kwargs = {} if hed_kwargs is None else hed_kwargs
        if hed:
            n_cats_per_dim = [len(self.search_space.params[param_name].categories) for param_name in
                          self.search_space.nominal_names]

            hed_num_embedders = self.hed_kwargs.get('hed_num_embedders', 128)
            kernel_dict["base_kernel_hed"] = HEDKernel(
                base_kernel=RBFKernel(), # base kernel irrelevant when using decompostions
                hed_num_embedders=hed_num_embedders,
                n_cats_per_dim=n_cats_per_dim,
                active_dims=search_space.nominal_dims
            )
        else:
            hed_num_embedders = 0

        hed_dims = [search_space.num_dims + i for i in range(hed_num_embedders)]

        numeric_dims = []
        for p in range(len(search_space.params) + len(hed_dims)):
            if p in search_space.cont_dims or p in search_space.disc_dims or p in hed_dims:
                numeric_dims.append(p)

        self.numeric_singletons = []
        self.nominal_singletons = []

        self.all_numeric_cliques = []
        self.all_nominal_cliques = []
        self.mixed_cliques = []

        self.clique_to_scale_ix = {}
        self.dim_to_lengthscale_ix = {}

        if base_kernel_num == "rbf":
            base_kernel_num_class = RBFKernel
        elif base_kernel_num == "mat52":
            base_kernel_num_class = MaternKernel
        else:
            raise NotImplementedError

        if base_kernel_nom == 'overlap':
            base_kernel_nom_class = Overlap
        elif base_kernel_nom == "transformed_overlap":
            base_kernel_nom_class = TransformedOverlap
        elif hed:  # no need to have a nominal kernel
            base_kernel_nom_class = Overlap
        else:
            raise NotImplementedError(base_kernel_nom)

        if num_lengthscale_constraint is None:
            num_lengthscale_constraint = Positive()
        if nom_lengthscale_constraint is None:
            nom_lengthscale_constraint = Positive()

        if base_kernel_kwargs_num is None:
            base_kernel_kwargs_num = {}
        if base_kernel_kwargs_nom is None:
            base_kernel_kwargs_nom = {}

        kernel_dict["numeric_singleton"] = ScaleKernel(base_kernel_num_class(ard_num_dims=1), **base_kernel_kwargs_num)
        kernel_dict["numeric_tuple"] = ScaleKernel(base_kernel_num_class(ard_num_dims=2), **base_kernel_kwargs_num)
        kernel_dict["nominal_singleton"] = ScaleKernel(base_kernel_nom_class(ard_num_dims=1),
                                                       lengthscale_constraint=nom_lengthscale_constraint,
                                                       **base_kernel_kwargs_nom)

        for ix, c in enumerate(numeric_dims):
            self.dim_to_lengthscale_ix[c] = ix

        scale_ix = 0
        for component in decomposition:

            component = tuple(sorted(component))
            if len(component) == 1:
                self.clique_to_scale_ix[component[0]] = scale_ix
                scale_ix += 1
                if component[0] in numeric_dims:
                    self.numeric_singletons.append(component[0])
                else:
                    self.nominal_singletons.append(component[0])
            else:
                if all(c in numeric_dims for c in component):
                    self.clique_to_scale_ix[component] = scale_ix
                    scale_ix += 1
                    self.all_numeric_cliques.append(component)

                elif not any(c in numeric_dims for c in component):
                    self.all_nominal_cliques.append(component)
                    kernel_dict[str(component)] = ScaleKernel(
                        base_kernel_nom_class(ard_num_dims=len(component), active_dims=component,
                                              lengthscale_constraint=nom_lengthscale_constraint,
                                              **base_kernel_kwargs_nom))

                else:
                    num_dims = [c for c in component if c in numeric_dims]
                    nom_dims = [c for c in component if c not in numeric_dims]
                    numerical_kernel = ScaleKernel(
                        base_kernel_num_class(ard_num_dims=len(num_dims), active_dims=tuple(sorted(num_dims)),
                                              **base_kernel_kwargs_num))
                    nominal_kernel = ScaleKernel(
                        base_kernel_nom_class(ard_num_dims=len(nom_dims), active_dims=tuple(sorted(nom_dims)),
                                              lengthscale_constraint=nom_lengthscale_constraint,
                                              **base_kernel_kwargs_nom))

                    component_kernel = MixtureKernel(search_space=search_space, numeric_kernel=numerical_kernel,
                                                     categorical_kernel=nominal_kernel)

                    self.mixed_cliques.append(component)
                    kernel_dict[str(component)] = component_kernel

        super(DecompositionKernel, self).__init__(ard_num_dims=search_space.num_dims,
                                                  lengthscale_constraint=num_lengthscale_constraint)
        self.kernel_dict = torch.nn.ModuleDict(kernel_dict)
        self.numeric_dims = numeric_dims
        self.hed_dims = hed_dims

        # create lengthscale for each numerical dimension
        self.has_lengthscale = True
        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()
        lengthscale_num_dims = len(numeric_dims)
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
        )
        if lengthscale_prior is not None:
            if not isinstance(lengthscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
            self.register_prior(
                "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
            )

        self.register_constraint("raw_lengthscale", lengthscale_constraint)

        # create outputscale for each batched component
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        self.register_parameter(name="raw_outputscale",
                                parameter=torch.nn.Parameter(torch.ones(len(self.clique_to_scale_ix.keys()))))
        self.register_constraint("raw_outputscale", outputscale_constraint)
        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(outputscale_prior).__name__)
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        self.base_kernel_num = base_kernel_num
        self.base_kernel_nom = base_kernel_nom

        self.hed = hed

        if self.hed:
            assert len(self.nominal_singletons) == 0
            assert len(self.all_nominal_cliques) == 0
            assert len(self.mixed_cliques) == 0

    @property
    def outputscale(self) -> torch.tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: Union[float, torch.tensor]):
        self._set_outputscale(value)

    def _set_outputscale(self, value: Union[float, torch.tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1: torch.tensor, x2: torch.tensor, diag: bool = False, last_dim_is_batch: bool = False,
                clique: tuple = None, **params) -> torch.tensor:
        if last_dim_is_batch:
            raise RuntimeError("DecompositionKernel does not accept the last_dim_is_batch argument.")

        if self.hed:
            x1_emb = self.kernel_dict["base_kernel_hed"].embed(x1[::, self.search_space.nominal_dims])
            x2_emb = self.kernel_dict["base_kernel_hed"].embed(x2[::, self.search_space.nominal_dims])

            x1 = torch.cat([x1, x1_emb], dim=-1)
            x2 = torch.cat([x2, x2_emb], dim=-1)

        if clique is not None:
            return self.partial_forward(x1, x2, clique=clique, last_dim_is_batch=last_dim_is_batch, **params)

        x1_ = x1.clone()
        x2_ = x2.clone()

        x1_[::, self.numeric_dims] = x1[::, self.numeric_dims].div(self.lengthscale)
        x2_[::, self.numeric_dims] = x2[::, self.numeric_dims].div(self.lengthscale)

        total = None

        # Numeric singletons
        if len(self.numeric_singletons) > 0:
            x1_single_num = x1_[::, self.numeric_singletons]
            x2_single_num = x2_[::, self.numeric_singletons]

            res = self.kernel_dict["numeric_singleton"](x1_single_num, x2_single_num, diag=diag, last_dim_is_batch=True,
                                                        **params)
            numeric_singleton_scales = self.outputscale[[self.clique_to_scale_ix[c] for c in self.numeric_singletons]]
            res = res.mul(numeric_singleton_scales.view(*([-1, 1] if diag else [-1, 1, 1])))
            if total is None:
                total = res.sum(-2 if diag else -3)
            else:
                total += res.sum(-2 if diag else -3)

        # Numeric tuples
        if len(self.all_numeric_cliques) > 0:
            x1_tuple_num = x1_[::, self.all_numeric_cliques]
            x1_tuple_num = x1_tuple_num.transpose(-3, -2)
            x2_tuple_num = x2_[::, self.all_numeric_cliques]
            x2_tuple_num = x2_tuple_num.transpose(-3, -2)

            res = self.kernel_dict["numeric_tuple"](x1_tuple_num, x2_tuple_num, diag=diag, last_dim_is_batch=False,
                                                    **params)
            all_numeric_scales = self.outputscale[[self.clique_to_scale_ix[c] for c in self.all_numeric_cliques]]
            res = res.mul(all_numeric_scales.view(*([-1, 1] if diag else [-1, 1, 1])))
            if total is None:
                total = res.sum(-2 if diag else -3)
            else:
                total += res.sum(-2 if diag else -3)

        # Nominal singletons
        if len(self.nominal_singletons) > 0:
            x1_single_nom = x1_[::, self.nominal_singletons]
            x2_single_nom = x2_[::, self.nominal_singletons]

            res = self.kernel_dict["nominal_singleton"](x1_single_nom, x2_single_nom, diag=diag, last_dim_is_batch=True,
                                                        **params)
            nominal_singleton_scales = self.outputscale[[self.clique_to_scale_ix[c] for c in self.nominal_singletons]]
            res = res.mul(nominal_singleton_scales.view(*([-1, 1] if diag else [-1, 1, 1])))
            if total is None:
                total = res.sum(-2 if diag else -3)
            else:
                total += res.sum(-2 if diag else -3)

        # Mixed and nominal tuples (for now non-batched and slow)
        for component in self.all_nominal_cliques + self.mixed_cliques:
            if total is None:
                total = self.kernel_dict[str(tuple(component))](x1_, x2_)
            else:
                total += self.kernel_dict[str(tuple(component))](x1_, x2_)

        return total

    def partial_forward(self, x1: torch.tensor, x2: torch.tensor, clique: Union[Tuple[int], List[int]],
                        diag: bool = False, last_dim_is_batch: bool = False, **params) -> torch.tensor:
        if last_dim_is_batch:
            raise RuntimeError("DecompositionKernel does not accept the last_dim_is_batch argument.")

        x1_ = x1.clone()
        x2_ = x2.clone()

        clique = tuple(sorted(clique))
        num_dims_in_clique = []
        ls = []
        for d in clique:
            if d in self.numeric_dims:
                num_dims_in_clique.append(d)
                ls.append(self.dim_to_lengthscale_ix[d])

        if len(num_dims_in_clique) > 0:
            x1_[::, num_dims_in_clique] /= self.lengthscale[:, ls]
            x2_[::, num_dims_in_clique] /= self.lengthscale[:, ls]

        # Singletons
        if len(clique) == 1:
            x1_single_num = x1_[::, clique]
            x2_single_num = x2_[::, clique]

            if clique[0] in self.numeric_singletons:
                res = self.kernel_dict["numeric_singleton"](x1_single_num, x2_single_num, diag=diag,
                                                            last_dim_is_batch=False, **params)

            if clique[0] in self.nominal_singletons:
                res = self.kernel_dict["nominal_singleton"](x1_single_num, x2_single_num, diag=diag,
                                                            last_dim_is_batch=False, **params)

            res *= self.outputscale[self.clique_to_scale_ix[clique[0]]]
            return res

        # Numeric tuples
        if clique in self.all_numeric_cliques:
            x1_tuple_num = x1_[::, clique]
            x2_tuple_num = x2_[::, clique]

            res = self.kernel_dict["numeric_tuple"](x1_tuple_num, x2_tuple_num, diag=diag, last_dim_is_batch=False,
                                                    **params)
            res *= self.outputscale[self.clique_to_scale_ix[clique]]

            return res

        # Mixed and nominal (for now non-batched and slow)
        if clique in self.all_nominal_cliques + self.mixed_cliques:
            return self.kernel_dict[str(tuple(sorted(clique)))](x1_, x2_)

        raise ValueError(f"Clique {clique} not in decomposition.")

    def get_lengthcales_numerical_dims(self) -> torch.tensor:
        return self.lengthscale[::, :self.lengthscale.shape[-1] - len(self.hed_dims)]

    def train(self, mode=True):
        self.kernel_dict.train(mode)