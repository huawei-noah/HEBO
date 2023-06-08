# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


# Implementation of various kernels
from typing import Union, Optional, List

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, NewtonGirardAdditiveKernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
import numpy as np


class MixtureKernel(Kernel):
    """
    The implementation of the mixed categorical and continuous kernel first proposed in CoCaBO, but re-implemented
    in gpytorch.

    Note that gpytorch uses the pytorch autodiff engine, and there is no need to manually define the derivatives of
    the kernel hyperparameters w.r.t the log-marginal likelihood as in the gpy implementation.
    """
    has_lengthscale = True

    def __init__(self, categorical_dims,
                 continuous_dims,
                 integer_dims=None,
                 lamda=0.5,
                 categorical_kern_type='transformed_overlap',
                 continuous_kern_type='mat52',
                 categorical_lengthscale_constraint=None,
                 continuous_lengthscale_constraint=None,
                 categorical_ard=True,
                 continuous_ard=True,
                 **kwargs):
        """

        Parameters
        ----------
        categorical_dims: the dimension indices that are categorical/discrete
        continuous_dims: the dimension indices that are continuous
        integer_dims: the **continuous indices** that additionally require integer constraint.
        lamda: \in [0, 1]. The trade-off between product and additive kernels. If this argument is not supplied, then
            lambda will be optimised as if it is an additional kernel hyperparameter
        categorical_kern_type: 'overlap', 'type2'
        continuous_kern_type: 'rbf' or 'mat52' (Matern 5/2)
        categorical_lengthscale_constraint: if supplied, the constraint on the lengthscale of the categorical kernel
        continuous_lengthscale_constraint: if supplied the constraint on the lengthscale of the continuous kernel
        categorical_ard: bool: whether to use Automatic Relevance Determination (ARD) for categorical dimensions
        continuous_ard: bool: whether to use ARD for continuous dimensions
        kwargs: additional parameters.
        """
        super(MixtureKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.optimize_lamda = lamda is None
        self.fixed_lamda = lamda if not self.optimize_lamda else None
        self.categorical_dims = categorical_dims
        self.continuous_dims = continuous_dims
        if integer_dims is not None:
            integer_dims_np = np.asarray(integer_dims).flatten()
            cont_dims_np = np.asarray(self.continuous_dims).flatten()
            if not np.all(np.in1d(integer_dims_np, cont_dims_np)):
                raise ValueError("if supplied, all continuous dimensions with integer constraint must be themselves "
                                 "contained in the continuous_dimensions!")
            # Convert the integer dims in terms of indices of the continuous dims
            integer_dims = np.where(np.in1d(self.continuous_dims, integer_dims))[0]

        self.register_parameter(name='raw_lamda', parameter=torch.nn.Parameter(torch.ones(1)))
        # The lambda must be between 0 and 1.
        self.register_constraint('raw_lamda', Interval(0., 1.))

        # Initialise the kernel on categorical dimensions
        if categorical_kern_type == 'overlap':
            self.categorical_kern = CategoricalOverlap(lengthscale_constraint=categorical_lengthscale_constraint,
                                                       ard_num_dims=len(categorical_dims) if categorical_ard else None)
        elif categorical_kern_type == 'transformed_overlap':
            self.categorical_kern = TransformedCategorical(lengthscale_constraint=categorical_lengthscale_constraint,
                                                           ard_num_dims=len(
                                                               categorical_dims) if categorical_ard else None)
        elif categorical_kern_type == 'additive':
            self.categorical_kern = NewtonGirardAdditiveKernel(
                base_kernel=TransformedCategorical(),
                active_dims=categorical_dims,
                max_degree=kwargs.get('additive_kernel_max_degree'),
                num_dims=len(categorical_dims)
            )
        else:
            raise NotImplementedError("categorical kernel type %s is not implemented. " % categorical_kern_type)

        # By default, we use the Matern 5/2 kernel
        if continuous_kern_type == 'mat52':
            self.continuous_kern = WrappedMatern(nu=2.5, ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                                 integer_dims=integer_dims,
                                                 lengthscale_constraint=continuous_lengthscale_constraint)
        elif continuous_kern_type == 'rbf':
            self.continuous_kern = WrappedRBF(ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                              integer_dims=integer_dims,
                                              lengthscale_constraint=continuous_lengthscale_constraint)
        elif continuous_kern_type == 'additive':
            self.continuous_kern = NewtonGirardAdditiveKernel(
                base_kernel=MaternKernel(),
                active_dims=continuous_dims,
                max_degree=kwargs.get('additive_kernel_max_degree'),
                num_dims=len(continuous_dims)
            )

        else:
            raise NotImplementedError("continuous kernel type %s is not implemented. " % continuous_kern_type)

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

    def forward(self, x1, x2, diag=False,
                x1_cont=None, x2_cont=None, **params):
        """
        Note that here I also give options to pass the categorical and continuous inputs separately (instead of jointly)
        because the categorical dimensions will not be differentiable, and thus there would be problems when we optimize
        the acquisition function.

        When passed separately, x1 and x2 refer the categorical (non-differentiable) data, whereas x1_cont and x2_cont
        are the continuous (differentiable) data.
        Parameters
        ----------
        x1
        x2
        diag
        x1_cont
        x2_cont
        params

        Returns
        -------

        """
        if x1_cont is None and x2_cont is None:
            assert x1.shape[-1] == len(self.categorical_dims) + len(self.continuous_dims), \
                'dimension mismatch. Expected number of dimensions %d but got %d in x1' % \
                (len(self.categorical_dims) + len(self.continuous_dims), x1.shape[1])
            x1_cont, x2_cont = x1[:, self.continuous_dims], x2[:, self.continuous_dims]
            # the categorical kernels are not differentiable w.r.t inputs, detach them to ensure the computing graph of
            # the autodiff engine is not broken.
            x1_cat, x2_cat = x1[:, self.categorical_dims].detach(), x2[:, self.categorical_dims].detach()
        else:
            assert x1.shape[1] == len(self.categorical_dims)
            assert x1_cont.shape[1] == len(self.continuous_dims)
            x1_cat, x2_cat = x1, x2
        # same in cocabo.
        cat_k = self.categorical_kern.forward(x1_cat, x2_cat, diag, **params)
        cont_k = self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)
        comb_k = (1. - self.lamda) * (cat_k + cont_k) + self.lamda * cat_k * cont_k
        return comb_k


def wrap(x1, x2, integer_dims):
    """The wrapping transformation for integer dimensions according to Garrido-Merchán and Hernández-Lobato (2020)."""
    if integer_dims is not None:
        for i in integer_dims:
            x1[:, i] = torch.round(x1[:, i])
            x2[:, i] = torch.round(x2[:, i])
    return x1, x2


class WrappedMatern(MaternKernel):
    """Matern kernels wrapped integer type of inputs according to
    Garrido-Merchán and Hernández-Lobato in
    "Dealing with Categorical and Integer-valued Variables in Bayesian Optimization with Gaussian Processes"

    Note: we deal with the categorical-valued variables using the kernels specifically used to deal with
    categorical variables (instead of the one-hot transformation).
    """

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedMatern, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class WrappedRBF(RBFKernel):
    """Similar to above, but applied to RBF."""

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedRBF, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        input_dim = x1.shape[1]

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        # First, convert one-hot to ordinal representation
        # diff = x1[:, None] - x2[None, :]
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).to(float)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / input_dim
        if diag:
            return torch.diagonal(k_cat, dim1=-1, dim2=-2).to(float)
        return k_cat.to(float)


class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        input_dim = x1.shape[1]

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        # diff = x1[:, None] - x2[None, :]
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        diff[torch.abs(diff) > 1e-5] = 1
        diff1 = torch.logical_not(diff).to(float)

        def rbf(d, ard):
            if ard:
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / input_dim)

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diagonal(k_cat, dim1=-2, dim2=-1)
        return k_cat.to(float)


class OrdinalKernel(Kernel):
    """
    The ordinal version of TransformedCategorical2 kernel (replace the Kronecker delta with
    the distance metric).
    config: the number of vertices per dimension
    """
    has_lengthscale = True

    def __init__(self, config, **kwargs):
        super(OrdinalKernel, self).__init__(**kwargs)
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config).view(-1)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # expected x1 and x2 are of shape N x D respectively
        diff = (x1[:, None] - x2[None, :]) / self.config.to(x1.device)
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
            return torch.diag(k_cat).to(float)
        return k_cat.to(float)
