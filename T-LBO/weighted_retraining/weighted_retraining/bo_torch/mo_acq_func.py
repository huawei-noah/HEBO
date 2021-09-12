# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Optional, Union, Iterable, Dict

import torch
from botorch.acquisition import qExpectedImprovement, MCAcquisitionObjective, IdentityMCObjective, \
    qUpperConfidenceBound
from botorch.exceptions import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from botorch.utils.transforms import concatenate_pending_points, normalize
from torch import Tensor


class qErrorAwareEI(qExpectedImprovement):

    def __init__(
            self,
            model: SingleTaskGP,
            error_model: SingleTaskGP,
            best_f: Union[float, Tensor],
            gamma: Union[float, str],
            eps: float,
            sampler: Optional[MCSampler] = None,
            error_sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = IdentityMCObjective(),
            error_objective: Optional[MCAcquisitionObjective] = IdentityMCObjective(),
            X_pending: Optional[Tensor] = None,
            configuration: Optional[str] = 'ratio',
            y_var_bounds: Optional[Tensor] = None,
            r_var_bounds: Optional[Tensor] = None,
            **kwargs: Dict[str, Any]
    ):
        super().__init__(model, best_f, sampler, objective, X_pending, **kwargs)
        # self.error_model = error_model
        self.add_module("error_model", error_model)

        self.gamma = gamma  # cost-aware parameter
        self.eps = eps      # add this to denominator for stability
        self.configuration = configuration

        if error_sampler is None:
            error_sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.add_module("error_sampler", error_sampler)

        if error_objective is None:
            if error_model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(error_objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "Only objectives of type MCAcquisitionObjective are supported for "
                "MC acquisition functions."
            )
        self.add_module("error_objective", error_objective)
        self.return_acqf_and_error = False

        self.y_var_bounds = y_var_bounds
        self.r_var_bounds = r_var_bounds

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Iterable[Tensor]:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior_obj_val = self.model.posterior(X)
        samples_obj_val = self.sampler(posterior_obj_val)
        obj_val = self.objective(samples_obj_val)
        obj_val = (obj_val - self.best_f.unsqueeze(-1)).clamp_min(0)

        posterior_error = self.error_model.posterior(X)
        samples_error = self.error_sampler(posterior_error)
        error_val: Tensor = self.error_objective(samples_error)
        # make sure predicted errors are positive (and avoid getting nan in gradient as we have error^gamma)
        error_val_pos: Tensor = error_val.clamp_min(1e-10)

        assert error_val_pos.shape == obj_val.shape, (error_val_pos.shape, obj_val.shape)
        obj_val_term = obj_val.max(dim=-1)[0].unsqueeze(-1)

        if isinstance(self.gamma, str):
            # use the posterior variance as the param. tuning the importance of the error prediction in the acqf.
            if self.gamma == 'post_obj_var':
                var = posterior_obj_val.variance.view(-1, 1).add_(1e-3)
                var_rescaled = var.div(var.max())
                gamma = var_rescaled
            elif self.gamma == 'post_obj_inv_var':
                var = posterior_obj_val.variance.view(-1, 1).add_(1e-3)
                var_rescaled = var.div(var.max())
                gamma = 1 / var_rescaled
            elif self.gamma == 'post_err_var':
                var = posterior_error.variance.view(-1, 1).add_(1e-3)
                gamma = var.div(var.max())
            elif self.gamma == 'post_err_inv_var':
                var = posterior_error.variance.view(-1, 1).add_(1e-3)
                gamma = 1 / var.div(var.max())
            elif self.gamma == 'post_min_var':
                obj_var = posterior_obj_val.variance.view(-1, 1).detach()
                err_var = posterior_error.variance.view(-1, 1).detach()
                assert self.y_var_bounds is not None, "Need to provide bounds to normalize obj targets for var computations."
                assert self.r_var_bounds is not None, "Need to provide bounds to normalize err targets for var computations."
                obj_var_normalized = normalize(obj_var, self.y_var_bounds)
                err_var_normalized = normalize(err_var, self.r_var_bounds)
                cat_var = torch.hstack([obj_var_normalized, err_var_normalized])
                gamma = cat_var.min(dim=1, keepdim=True)[0]
            elif self.gamma == 'post_var_tradeoff':
                obj_var = posterior_obj_val.variance.view(-1, 1).detach()
                err_var = posterior_error.variance.view(-1, 1).detach()
                assert self.y_var_bounds is not None, "Need to provide bounds to normalize obj targets for var computations."
                assert self.r_var_bounds is not None, "Need to provide bounds to normalize err targets for var computations."
                obj_var_normalized = normalize(obj_var, self.y_var_bounds)
                err_var_normalized = normalize(err_var, self.r_var_bounds).add_(1e-3)
                gamma = obj_var_normalized.div(err_var_normalized)
            elif self.gamma == 'post_var_inv_tradeoff':
                obj_var = posterior_obj_val.variance.view(-1, 1).detach()
                err_var = posterior_error.variance.view(-1, 1).detach()
                assert self.y_var_bounds is not None, "Need to provide bounds to normalize obj targets for var computations."
                assert self.r_var_bounds is not None, "Need to provide bounds to normalize err targets for var computations."
                obj_var_normalized = normalize(obj_var, self.y_var_bounds).add_(1e-3)
                err_var_normalized = normalize(err_var, self.r_var_bounds)
                gamma = err_var_normalized.div(obj_var_normalized)
            else:
                raise NotImplementedError(f"gamma setting not defined: {self.gamma}")
        else:
            assert isinstance(self.gamma, float) or isinstance(self.gamma, int), self.gamma
            gamma = self.gamma

        if self.configuration == 'product':
            error_term = error_val_pos.pow(gamma)
            error_aware_obj_val = obj_val_term.mul(error_term).squeeze()
        else:
            assert self.configuration == 'ratio', self.configuration
            error_term = error_val_pos.mul(self.eps).add(1).pow(gamma)
            error_aware_obj_val = obj_val_term.div(error_term).squeeze()

        if self.return_acqf_and_error:
            return error_aware_obj_val.mean(dim=0), obj_val_term.mean(dim=0), error_val_pos.mean(dim=0)
        else:
            return error_aware_obj_val.mean(dim=0)


class qErrorAwareUCB(qUpperConfidenceBound):
    def __init__(
            self,
            model: SingleTaskGP,
            error_model: SingleTaskGP,
            beta: float,
            gamma: Union[float, str],
            eps: float,
            sampler: Optional[MCSampler] = None,
            error_sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = IdentityMCObjective(),
            error_objective: Optional[MCAcquisitionObjective] = IdentityMCObjective(),
            X_pending: Optional[Tensor] = None,
            configuration: Optional[str] = 'ratio',
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super(qErrorAwareUCB, self).__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending, beta=beta
        )
        self.add_module("error_model", error_model)

        self.gamma = gamma  # cost-aware parameter
        self.eps = eps  # add this to denominator for stability
        self.configuration = configuration

        if error_sampler is None:
            error_sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.add_module("error_sampler", error_sampler)

        if error_objective is None:
            if error_model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(error_objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "Only objectives of type MCAcquisitionObjective are supported for "
                "MC acquisition functions."
            )
        self.add_module("error_objective", error_objective)
        self.return_acqf_and_error = False

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Iterable[Tensor]:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior_obj_val = self.model.posterior(X)
        samples_obj_val = self.sampler(posterior_obj_val)
        obj_val = self.objective(samples_obj_val)
        obj_val_mean = obj_val.mean(dim=0)
        ucb_samples = obj_val_mean + self.beta_prime * (obj_val - obj_val_mean).abs()
        obj_val_term = ucb_samples.max(dim=-1)[0].unsqueeze(-1)

        posterior_error = self.error_model.posterior(X)
        samples_error = self.error_sampler(posterior_error)
        error_val: Tensor = self.error_objective(samples_error)
        # make sure predicted errors are positive (and avoid getting nan in gradient as we have error^gamma)
        error_val_pos: Tensor = error_val.clamp_min(1e-10)

        assert error_val_pos.shape == ucb_samples.shape, (error_val_pos.shape, ucb_samples.shape)

        if isinstance(self.gamma, str):
            # use the posterior variance as the param. tuning the importance of the error prediction in the acqf.
            if self.gamma == 'post_obj_var':
                var = posterior_obj_val.variance.view(-1, 1)
                var_rescaled = var.div(var.max())
                gamma = 1 / var_rescaled
            else:
                gamma = posterior_error.variance.view(-1, 1)
        else:
            assert isinstance(self.gamma, float)
            gamma = self.gamma

        if self.configuration == 'product':
            error_term = error_val_pos.pow(gamma)
            error_aware_obj_val = obj_val_term.mul(error_term).squeeze()
        else:
            assert self.configuration == 'ratio', self.configuration
            error_term = error_val_pos.mul(self.eps).add(1).pow(gamma)
            error_aware_obj_val = obj_val_term.div(error_term).squeeze()

        if self.return_acqf_and_error:
            return error_aware_obj_val.mean(dim=0), obj_val_term.mean(dim=0), error_val_pos.mean(dim=0)
        else:
            return error_aware_obj_val.mean(dim=0)
