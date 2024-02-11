"""
uGP implementation
"""
from models.mmd_gp import MMDGP
from kernels.kme_kernel import KMEKernel
import gpytorch as gpyt


class UGP(MMDGP):
    def __init__(self, train_inputs, train_targets, likelihood, num_inputs,
                 input_transform=None, outcome_transform=None, additional_transform=None,
                 hidden_dims=(4, 2), latent_dim=1, **kwargs):
        super(UGP, self).__init__(train_inputs, train_targets, likelihood, num_inputs,
                                        input_transform, outcome_transform,
                                        additional_transform, **kwargs)

    def define_covar_module(self, **kwargs):
        xc_kern_params = {}
        xc_lscale_constr = kwargs.get('xc_ls_constr', None)
        if xc_lscale_constr is not None:
            xc_kern_params['lengthscale_constraint'] = xc_lscale_constr

        xc_kme_inner_k = kwargs.get('base_kernel', None)
        if xc_kme_inner_k is None:
            xc_kme_inner_k = gpyt.kernels.RBFKernel(**xc_kern_params)

        estimator_name = kwargs.get('estimator_name', 'integral')
        chunk_size = kwargs.get('chunk_size', 100)
        sub_samp_size = kwargs.get('sub_samp_size', 100)
        covar_module = gpyt.kernels.ScaleKernel(
            KMEKernel(xc_kme_inner_k, estimator=estimator_name, chunk_size=chunk_size)
        )
        return covar_module