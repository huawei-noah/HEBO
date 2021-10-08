import math
from typing import Optional, List
import numpy as np

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.functions import MaternCovariance
from gpytorch.kernels import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.settings import trace_mode
from torch import Tensor

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling

MIN_INFERRED_NOISE_LEVEL = 1e-4


class SingleTaskRoundGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""A single-task exact GP model.

    A single-task exact GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the
    same training data. If outputs are independent and outputs have different
    training data, use the ModelListGP. When modeling correlations between
    outputs, use the MultiTaskGP.
    """

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            normalization_offset: Tensor,
            normalization_scale: Tensor,
            int_mask: Optional[List[int]] = None,
            likelihood: Optional[Likelihood] = None,
            covar_module: Optional[Module] = None,
            outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        r"""A single-task exact GP model.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            int_mask: List of indices of X variables that are integer-valued (if not specified, consider all entries as integers)
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> model = SingleTaskRoundGP(train_X, train_Y)
        """
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=train_X, Y=train_Y)
        validate_input_scaling(train_X=train_X, train_Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(self, train_X, train_Y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternRoundedKernel(
                    nu=2.5,
                    normalization_offset=normalization_offset,
                    normalization_scale=normalization_scale,
                    int_mask=int_mask,
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self.covar_module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MaternRoundedKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Matern kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Matern}}(\mathbf{x_1}, \mathbf{x_2}) = \frac{2^{1 - \nu}}{\Gamma(\nu)}
          \left( \sqrt{2 \nu} d \right) K_\nu \left( \sqrt{2 \nu} d \right)
       \end{equation*}

    where

    * :math:`d = (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-1} (\mathbf{x_1} - \mathbf{x_2})`
      is the distance between
      :math:`x_1` and :math:`x_2` scaled by the :attr:`lengthscale` parameter :math:`\Theta`.
    * :math:`\nu` is a smoothness parameter (takes values 1/2, 3/2, or 5/2). Smaller values are less smooth.
    * :math:`K_\nu` is a modified Bessel function.

    There are a few options for the lengthscale parameter :math:`\Theta`:
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`nu` (float):
            The smoothness parameter: either 1/2, 3/2, or 5/2.
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `None`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to
            compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.
        :attr:'int_mask' (List[int]):
            List of indices of variables that are integer-valued (if not specified, consider all entries as integers)

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([2])
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def __init__(self, normalization_offset: Tensor, normalization_scale: Tensor, nu=2.5,
                 int_mask: Optional[List[int]] = None, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternRoundedKernel, self).__init__(**kwargs)
        self.nu = nu
        aux_int_mask = np.arange(self.ard_num_dims) if int_mask is None else int_mask
        self.normalization_offset = normalization_offset[aux_int_mask]
        self.normalization_scale = normalization_scale[aux_int_mask]
        self.int_mask = int_mask

    def forward(self, x1: Tensor, x2: Tensor, diag=False, **params):

        if (
                x1.requires_grad
                or x2.requires_grad
                or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                or diag
                or trace_mode.on()
        ):
            if self.int_mask is not None:
                x1[..., self.int_mask] = x1[..., self.int_mask].add(-self.normalization_offset).div(
                    self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
                x2[..., self.int_mask] = x2[..., self.int_mask].add(-self.normalization_offset).div(
                    self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
            else:
                x1 = x1.add(-self.normalization_offset).div(
                    self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
                x2 = x2.add(-self.normalization_offset).div(
                    self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component

        x1 = x1.clone()
        x2 = x2.clone()
        if self.int_mask is not None:
            x1[..., self.int_mask] = x1[..., self.int_mask].add(-self.normalization_offset).div(
                self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
            x2[..., self.int_mask] = x2[..., self.int_mask].add(-self.normalization_offset).div(
                self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
        else:
            x1 = x1.add(-self.normalization_offset).div(
                self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
            x2 = x2.add(-self.normalization_offset).div(
                self.normalization_scale).round().mul(self.normalization_scale).add(self.normalization_offset)
        return MaternCovariance().apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )
