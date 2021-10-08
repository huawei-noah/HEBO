from typing import Union, Optional

import torch
from botorch.acquisition import qExpectedImprovement, MCAcquisitionObjective, qProbabilityOfImprovement, qSimpleRegret, \
    qUpperConfidenceBound
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils import draw_sobol_normal_samples
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform

from core.comp_acquisition.compositional_acquisition import CompositionalAcquisition
from torch import Tensor


class qCompositionalExpectedImprovement(qExpectedImprovement, CompositionalAcquisition):
    """MC-based batch Expected Improvement in compositional form. """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            m: int,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
            K_g: int = 64,
            fixed_z: bool = False,
    ) -> None:
        """q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: sampler that can be used to sample from posterior of the model
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
            K_g: number of inner samples used at each optimization step
            fixed_z: whether to use fixed z samples across optimization steps (set to `False` for memory efficient (ME))
            m: number of z samples considered to build `g` (should be equal to `K_g` for `ME` version)
        """
        super(qCompositionalExpectedImprovement, self).__init__(
            model=model, best_f=best_f, sampler=sampler, objective=objective, X_pending=X_pending
        )
        CompositionalAcquisition.__init__(self, fixed_z=fixed_z,
                                          K_g=K_g, m=m)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_expected(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()
        samples_z = draw_sobol_normal_samples(q, m, dtype=X.dtype, device=X.device).permute(1, 0)
        assert samples_z.shape == (q, m), (samples_z.shape, (q, m))
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        assert mu.shape == (t_batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        assert L.shape == (t_batch, q, q)
        samples = mu + L.matmul(samples_z)  # shape (t-batch, q, Kt_g)
        assert samples.shape == (t_batch, q, m)

        g_X: Tensor = samples - self.best_f
        assert g_X.shape == (t_batch, q, m), f"Expected shape {(t_batch, q, m)}, got {g_X.shape}"
        return g_X.div(m)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_oracle(self, X: Tensor, custom_z_filter: Optional[Tensor] = None) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        if custom_z_filter is not None:
            z_filter = custom_z_filter
            Kt_g = z_filter.sum()
        else:
            z_filter = self.z_filter
            Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z) - self.best_f  # shape (t_batch, q, K_g)
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        return g_X.mul_(1 / Kt_g).to(X)

    def outer_f(self, Y: Tensor) -> Tensor:
        assert Y.ndim == 3
        t_batch, q, n = Y.shape
        assert n == self.Kt_g, (n, self.Kt_g)
        f_Y = Y.clamp_min(0).max(dim=-2)[0]
        return f_Y.sum(dim=-1)  #

    def nested_eval(self, X: Tensor, **kwargs) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        z_filter = self.z_filter
        Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z) - self.best_f  # shape (t_batch, q, K_g)
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        return g_X.mul_(1 / Kt_g).clamp_min(0).max(dim=-2)[0].sum(dim=-1).to(X)


class qCompositionalProbabilityOfImprovement(qProbabilityOfImprovement, CompositionalAcquisition):
    """ MC-based batch Probability of Improvement in compositional form. """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            m: int,
            sampler: Optional[MCSampler] = None,
            X_pending: Optional[Tensor] = None,
            tau: float = 1e-3,
            K_g: int = 64,
            fixed_z: bool = False,
    ) -> None:
        """q-Probability of Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can
                be a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: sampler that can be used to sample from posterior of the model
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
            K_g: number of inner samples used at each optimization step
            fixed_z: whether to use fixed z samples across optimization steps (set to `False` for memory efficient (ME))
            m: number of z samples considered to build `g` (should be equal to `K_g` for `ME` version)
        """
        super(qCompositionalProbabilityOfImprovement, self).__init__(model, best_f, sampler, None, X_pending, tau)
        CompositionalAcquisition.__init__(self, fixed_z=fixed_z, K_g=K_g, m=m)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_oracle(self, X: Tensor, custom_z_filter: Optional[Tensor] = None) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        if custom_z_filter is not None:
            z_filter = custom_z_filter
            Kt_g = z_filter.sum()
        else:
            z_filter = self.z_filter
            Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z) - self.best_f
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape
        return g_X.mul_(1 / Kt_g).to(X)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_expected(self, X: Tensor) -> Tensor:
        t_batch, q, d = X.shape
        m: int = self.get_m()

        # Samples for z and omega are common to all t_batches

        posterior = self.model.posterior(X)
        samples_z = draw_sobol_normal_samples(q, m, dtype=X.dtype, device=X.device).permute(1, 0)
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        samples = mu + L.matmul(samples_z)  # shape (t-batch, q, Kt_g)
        assert samples.shape == (t_batch, q, m)

        g_X: Tensor = samples
        assert g_X.shape == (t_batch, q, m), f"Expected shape {(t_batch, q, m)}, got {g_X.shape}"
        return g_X.add(-self.best_f).div(m)

    def outer_f(self, Y: Tensor) -> Tensor:
        t_batch, q, n = Y.shape
        assert n == self.Kt_g, (n, self.Kt_g)
        f_Y = torch.sigmoid(Y.max(dim=-2)[0].mul(self.get_m()) / self.tau)  # shape t_batch x m
        assert f_Y.shape == (t_batch, n)
        return f_Y.sum(dim=-1).true_divide(self.get_m())

    def nested_eval(self, X: Tensor, **kwargs) -> Tensor:
        r""" Evaluate the acquisition function computing f(\hat{g}) where \hat{g} is an estimator of E_w[g_w(x)]
        Args:
            X: input
            smooth: bool
                whether to use exact formulation of PI or the smooth approximation (sigmoid with parameter `tau`)
        """
        smooth: bool = kwargs['smooth']
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        z_filter = self.z_filter
        Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z) - self.best_f
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape
        if not smooth:
            f_g_X = g_X.max(dim=-2)[0] > 0
            return f_g_X.sum(dim=-1).true_divide(self.get_m())
        else:
            g_X = g_X.mul(1 / Kt_g).to(X)
            f_g_X = torch.sigmoid(g_X.max(dim=-2)[0].mul(self.get_m()) / self.tau)
            # account for empty columns in g(X)
            return f_g_X.sum(dim=-1).add(
                torch.sigmoid(torch.zeros(t_batch, m - Kt_g, device=X.device, dtype=X.dtype)).sum(dim=-1)).true_divide(
                self.get_m())


class qCompositionalSimpleRegret(qSimpleRegret, CompositionalAcquisition):
    """ MC-based batch Simple Regret in compositional form. """

    def __init__(
            self,
            model: Model,
            m: int,
            sampler: Optional[MCSampler] = None,
            X_pending: Optional[Tensor] = None,
            K_g: int = 1,
            fixed_z: bool = False,

    ) -> None:
        """q-Simple Regret.

        Args:
            model: A fitted model.
            sampler: sampler that can be used to sample from posterior of the model
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
            K_g: number of inner samples used at each optimization step
            fixed_z: whether to use fixed z samples across optimization steps (set to `False` for memory efficient (ME))
            m: number of z samples considered to build `g` (should be equal to `K_g` for `ME` version)
        """
        super(qCompositionalSimpleRegret, self).__init__(model, sampler, None, X_pending)
        CompositionalAcquisition.__init__(self, fixed_z=fixed_z, K_g=K_g, m=m)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_oracle(self, X: Tensor, custom_z_filter: Optional[Tensor] = None) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        if custom_z_filter is not None:
            z_filter = custom_z_filter
            Kt_g = z_filter.sum()
        else:
            z_filter = self.z_filter
            Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z)
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        return g_X.mul_(1 / Kt_g).to(X)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_expected(self, X: Tensor) -> Tensor:
        t_batch, q, d = X.shape
        m: int = self.get_m()

        # Samples for z and omega are common to all t_batches

        posterior = self.model.posterior(X)

        samples_z = draw_sobol_normal_samples(q, m, dtype=X.dtype, device=X.device).permute(1, 0)
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        samples = mu + L.matmul(samples_z)  # shape (t-batch, q, Kt_g)
        assert samples.shape == (t_batch, q, m)

        g_X: Tensor = samples
        assert g_X.shape == (t_batch, q, m), f"Expected shape {(t_batch, q, m)}, got {g_X.shape}"
        return g_X.div(m)

    def outer_f(self, Y: Tensor) -> Tensor:
        t_batch, q, n = Y.shape
        assert n == self.Kt_g, (n, self.Kt_g)
        f_Y = Y.max(dim=-2)[0]  # shape t_batch x m
        assert f_Y.shape == (t_batch, n)
        return f_Y.sum(dim=-1)

    def nested_eval(self, X: Tensor, **kwargs) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        z_filter = self.z_filter
        Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)

        g_X: Tensor = mu + L.matmul(samples_z)
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        return g_X.mul(1 / Kt_g).max(dim=-2)[0].sum(dim=-1).to(X)


class qCompositionalUpperConfidenceBound(qUpperConfidenceBound, CompositionalAcquisition):
    """ MC-based batch Upper Confidence Bound in compositional form. """

    def __init__(self, model: Model,
                 beta: float,
                 m: int,
                 sampler: Optional[MCSampler] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 X_pending: Optional[Tensor] = None,
                 K_g: int = 64,
                 fixed_z: bool = False,
                 ) -> None:
        """q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: sampler that can be used to sample from posterior of the model
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
            K_g: number of inner samples used at each optimization step
            fixed_z: whether to use fixed z samples across optimization steps (set to `False` for memory efficient (ME))
            m: number of z samples considered to build `g` (should be equal to `K_g` for `ME` version)
        """
        super(qCompositionalUpperConfidenceBound, self).__init__(model, beta, sampler, objective, X_pending)
        CompositionalAcquisition.__init__(self, fixed_z=fixed_z, K_g=K_g, m=m)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_expected(self, X: Tensor):
        t_batch, q, d = X.shape
        m: int = self.get_m()

        # Samples for z and omega are common to all t_batches
        posterior = self.model.posterior(X)

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)  # shape (q, m)
        # get portion of mu and L that we need given sampled omegas
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        assert L.shape == (t_batch, q, q)
        fact = self.get_m()
        g_X: Tensor = mu + self.beta_prime * (L.matmul(samples_z)).abs()
        assert g_X.shape == (t_batch, q, m)
        return g_X.div(fact)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def inner_g_oracle(self, X: Tensor, custom_z_filter: Optional[Tensor] = None) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        if custom_z_filter is not None:
            z_filter = custom_z_filter
            Kt_g = z_filter.sum().item()
        else:
            z_filter = self.z_filter
            Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        g_X: Tensor = mu + self.beta_prime * (L.matmul(samples_z)).abs()
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        return g_X.mul(1 / Kt_g).to(X)

    def outer_f(self, Y: Tensor) -> Tensor:
        t_batch, q, n = Y.shape
        assert n == self.Kt_g, (n, self.Kt_g)
        f_Y = Y.max(dim=-2)[0]  # shape t_batch x m
        assert f_Y.shape == (t_batch, n)

        # take mean of the K_f f_v for each t_batch (shape is (t_batch,))
        return f_Y.sum(dim=-1)

    def nested_eval(self, X: Tensor, **kwargs) -> Tensor:
        posterior = self.model.posterior(X)
        t_batch, q, d = X.shape
        m: int = self.get_m()

        z_filter = self.z_filter
        Kt_g = self.Kt_g

        samples_z: Tensor = self.z_samples(q, m, dtype=X.dtype, device=X.device)[:, z_filter]  # shape (q, Kt_g)
        assert samples_z.shape == (q, Kt_g), samples_z.shape
        mu: Tensor = posterior.mean  # shape (t-batch, q, 1)
        L: Tensor = posterior.mvn.lazy_covariance_matrix.cholesky(upper=False).evaluate()  # shape (t-batch, q, q)
        g_X: Tensor = mu + self.beta_prime * (L.matmul(samples_z)).abs()
        assert g_X.shape == (t_batch, q, Kt_g), g_X.shape

        g_X.mul_(1 / Kt_g).to(X)

        f_g_X = g_X.max(dim=-2)[0]  # shape t_batch x m

        # take mean of the K_f f_v for each t_batch (shape is (t_batch,))
        return f_g_X.sum(dim=-1)
