import math
from typing import Union, Optional

from botorch.acquisition import qExpectedImprovement, qSimpleRegret, qProbabilityOfImprovement, qUpperConfidenceBound, \
    MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling import MCSampler
import torch
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


class qFiniteSumExpectedImprovement(qExpectedImprovement):
    r"""MC-based batch Expected Improvement in FS form.
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            K_g: int,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Expected Improvement FS.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: sampler that can be used to sample from posterior of the model
            objective: The MCAcquisitionObjective under which the samples are evalauted.
                Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
            K_g: number of inner samples used at each optimization step
        """
        super().__init__(best_f=best_f,
                         model=model, sampler=sampler, objective=objective, X_pending=X_pending
                         )
        self.base_samples_z = None
        self.K_g = K_g

    def z_samples(self, *size, dtype, device=None) -> Tensor:
        if self.base_samples_z is None or self.base_samples_z.shape != (*size,):
            self.base_samples_z = torch.randn(*size, dtype=dtype, device=device)
        return self.base_samples_z

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        m = self.sampler.sample_shape.numel()
        posterior = self.model.posterior(X)

        z_inds = torch.randint(0, m, size=(self.K_g,))
        z_filter = torch.zeros(m, dtype=bool, device=X.device)
        z_filter[z_inds] = 1

        samples = self.z_samples(X.shape[-2], m, dtype=X.dtype, device=X.device)[:, z_filter]
        mean = posterior.mean
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        samples = mean + (L.matmul(samples))
        samples = samples.permute(2, 0, 1)
        obj = self.objective(samples)
        obj = (obj - self.best_f).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei


class qFiniteSumSimpleRegret(qSimpleRegret):
    r"""MC-based batch Simple Regret in FS form."""

    def __init__(
            self,
            model: Model,
            K_g: int,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        self.base_samples_z = None
        self.K_g = K_g

    def z_samples(self, *size, dtype, device=None) -> Tensor:
        if self.base_samples_z is None or self.base_samples_z.shape != (*size,):
            self.base_samples_z = torch.randn(*size, dtype=dtype, device=device)
        return self.base_samples_z

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qSimpleRegret on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Simple Regret values at the given design
            points `X`, where `batch_shape'` is the broadcasted batch shape of model
            and input `X`.
        """
        m = self.sampler.sample_shape.numel()
        posterior = self.model.posterior(X)

        z_inds = torch.randint(0, m, size=(self.K_g,))
        z_filter = torch.zeros(m, dtype=bool, device=X.device)
        z_filter[z_inds] = 1

        samples = self.z_samples(X.shape[-2], m, dtype=X.dtype, device=X.device)[:, z_filter]
        mean = posterior.mean
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        samples = mean + (L.matmul(samples))
        samples = samples.permute(2, 0, 1)

        obj = self.objective(samples)
        val = obj.max(dim=-1)[0].mean(dim=0)
        return val


class qFiniteSumProbabilityOfImprovement(qProbabilityOfImprovement):
    r"""MC-based batch Probability of Improvement in FS form."""

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            K_g: int,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
            tau: float = 1e-3,
    ) -> None:
        r"""q-Probability of Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can
                be a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: sampler that can be used to sample from posterior of the model
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
            K_g: number of inner samples used at each optimization step
        """
        super().__init__(best_f=best_f,
                         model=model, sampler=sampler, objective=objective, X_pending=X_pending
                         )
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(float(best_f))
        self.register_buffer("best_f", best_f)
        if not torch.is_tensor(tau):
            tau = torch.tensor(float(tau))
        self.register_buffer("tau", tau)

        self.base_samples_z = None
        self.K_g = K_g

    def z_samples(self, *size, dtype, device=None) -> Tensor:
        if self.base_samples_z is None or self.base_samples_z.shape != (*size,):
            self.base_samples_z = torch.randn(*size, dtype=dtype, device=device)
        return self.base_samples_z

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Probability of Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        m = self.sampler.sample_shape.numel()
        posterior = self.model.posterior(X)

        z_inds = torch.randint(0, m, size=(self.K_g,))
        z_filter = torch.zeros(m, dtype=bool, device=X.device)
        z_filter[z_inds] = 1

        samples = self.z_samples(X.shape[-2], m, dtype=X.dtype, device=X.device)[:, z_filter]
        mean = posterior.mean
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        samples = mean + (L.matmul(samples))
        samples = samples.permute(2, 0, 1)
        obj = self.objective(samples)
        max_obj = obj.max(dim=-1)[0]
        val = torch.sigmoid((max_obj - self.best_f) / self.tau).mean(dim=0)
        return val


class qFiniteSumUpperConfidenceBound(qUpperConfidenceBound):
    r"""MC-based batch Upper Confidence Bound in FS form.
    """

    def __init__(
            self,
            model: Model,
            beta: float,
            K_g: int,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(beta=beta,
                         model=model, sampler=sampler, objective=objective, X_pending=X_pending
                         )
        self.beta_prime = math.sqrt(beta * math.pi / 2)
        self.base_samples_z = None
        self.K_g = K_g

    def z_samples(self, *size, dtype, device=None) -> Tensor:
        if self.base_samples_z is None or self.base_samples_z.shape != (*size,):
            self.base_samples_z = torch.randn(*size, dtype=dtype, device=device)
        return self.base_samples_z

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        m = self.sampler.sample_shape.numel()
        posterior = self.model.posterior(X)

        z_inds = torch.randint(0, m, size=(self.K_g,))
        z_filter = torch.zeros(m, dtype=bool, device=X.device)
        z_filter[z_inds] = 1

        samples = self.z_samples(X.shape[-2], m, dtype=X.dtype, device=X.device)[:, z_filter]
        mean = posterior.mean
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        ucb_samples = mean + self.beta_prime * (L.matmul(samples)).abs()
        ucb_samples = ucb_samples.permute(2, 0, 1)
        return ucb_samples.max(dim=-1)[0].mean(dim=0)
