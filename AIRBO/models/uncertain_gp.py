from model_utils import input_transform as tfx
from gpytorch.kernels import GaussianSymmetrizedKLKernel, RBFKernel
from kernels.expected_rbf_kernel import ExpectedRBFKernel
from models.robust_gp import RobustGP

import gpytorch as gpyt
from botorch.models import transforms as tf
import torch

KN_EXPECTED_RBF = 'ERBF'
KN_SKL = "SKL"
KN_RBF = 'rbf'


class UncertainGP(RobustGP):
    def __init__(self, train_inputs, train_targets, likelihood, num_inputs,
                 input_transform=None, outcome_transform=None, additional_transform=None,
                 **kwargs):
        super(UncertainGP, self).__init__(train_inputs, train_targets, likelihood, num_inputs,
                                          input_transform, outcome_transform,
                                          additional_transform, **kwargs)
        self.kernel_name = kwargs.get('kernel_name', KN_SKL)

    def define_covar_module(self, **kwargs):
        n_var = kwargs['n_var']
        self.kernel_name = kwargs['kernel_name']
        xc_lscale_constr = kwargs.get('xc_lscale_constr', None)
        xc_kern_params = {'ard_num_dims': n_var}
        if xc_lscale_constr is not None:
            xc_kern_params['lengthscale_constraint'] = xc_lscale_constr

        if self.kernel_name == KN_EXPECTED_RBF:
            xc_kern_params['ard_num_dims'] = None  # ERBF kernel cannot use ARD
            xc_covar_module = gpyt.kernels.ScaleKernel(ExpectedRBFKernel(**xc_kern_params))
        elif self.kernel_name == KN_SKL:
            xc_kern_params['ard_num_dims'] = None  # SKL kernel cannot use ARD
            xc_covar_module = gpyt.kernels.ScaleKernel(
                GaussianSymmetrizedKLKernel(**xc_kern_params)
            )
        elif self.kernel_name == KN_RBF:
            xc_covar_module = gpyt.kernels.ScaleKernel(RBFKernel(**xc_kern_params))
        else:
            raise ValueError("Unsupported kernel type:", self.kernel_name)

        return xc_covar_module

    def define_default_input_transform(self, **kwargs):
        n_var = kwargs['n_var']
        input_bounds = kwargs['input_bounds']
        return tfx.MultiInputTransform(
            tf1=tf.Normalize(d=n_var, bounds=input_bounds, transform_on_train=True),
            tf2=tfx.ScaleTransform(d=n_var, bounds=input_bounds, transform_on_train=True),
            tf3=tfx.DummyTransform(transform_on_train=True),
        )

    def define_additional_transform(self, **kwargs):
        raw_input_std = kwargs['raw_input_std']
        return tfx.AdditionalFeatures(f=tfx.additional_std, transform_on_train=False,
                                      fkwargs={'std': raw_input_std})

    def compute_mean_cov(self, X, **kwargs):
        xc_raw, xc_std, xe = X
        mean_x, covar = None, None
        if self.kernel_name == KN_SKL:
            xc_input = torch.concat((xc_raw, xc_std.pow(2).log()), dim=-1)  # SKL takes log variance
        elif self.kernel_name == KN_EXPECTED_RBF:
            xc_input = torch.concat((xc_raw, xc_std.pow(2)), dim=-1)  # eRBF takes variance as input
        else:
            xc_input = xc_raw

        # input K
        mean_x = self.mean_module(xc_raw)
        covar_x = self.covar_module(xc_input, **kwargs)
        return mean_x, covar_x

    def forward(self, xc_raw, xc_std, xe):
        # note that we assume X is already applied with additional transform
        # input transform
        X = (xc_raw, xc_std, xe)
        if self.training:
            X = self.transform_inputs(X)
        mean_x, covar = self.compute_mean_cov(X)
        return gpyt.distributions.MultivariateNormal(mean_x, covar)
