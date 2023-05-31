# 2021.11.10-Specify `tr` in adjust_length
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from copy import deepcopy

import gpytorch
import math
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from resources.casmopolitan.bo.localbo_cat import CASMOPOLITANCat
from resources.casmopolitan.bo.localbo_utils import train_gp, random_sample_within_discrete_tr_ordinal


class CASMOPOLITANMixed(CASMOPOLITANCat):
    """

    Parameters
    ----------
    config: the configuration for the categorical dimensions
    cat_dims: the list of indices of dimensions that are categorical
    cont_dims: the list of indices of dimensions that are continuous
    *Note*: in general, you should have the first d_cat dimensions as cat_dims, and the rest as the continuous dims.
    lb : Lower variable bounds of the continuous dimensions, numpy.array, shape (d_cont,).
    ub : Upper variable bounds of the continuous dimensions, numpy.array, shape (d_cont,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")
    """

    def __init__(
            self,
            config,
            cat_dim,
            cont_dim,
            lb,
            ub,
            n_init,
            max_evals,
            batch_size=1,
            int_constrained_dims=None,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device="cpu",
            dtype="float32",
            acq='thompson',
            kernel_type='mixed',
            **kwargs
    ):
        super(CASMOPOLITANMixed, self).__init__(len(cat_dim) + len(cont_dim),
                                                n_init, max_evals, config, batch_size, verbose, use_ard,
                                                max_cholesky_size, n_training_steps, min_cuda,
                                                device, dtype, acq, kernel_type, **kwargs)
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)

        self.kwargs = kwargs
        # Save function information for both the continuous and categorical parts.
        self.cat_dims, self.cont_dims = cat_dim, cont_dim
        self.int_constrained_dims = int_constrained_dims
        # self.n_categories = n_cats
        self.lb = lb
        self.ub = ub

    def adjust_tr_length(self, fX_next):
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            # self.length = min([self.tr_multiplier * self.length, self.length_max])
            # For the Hamming distance-bounded trust region, we additively (instead of multiplicatively) adjust.
            self.length_discrete = int(min(self.length_discrete * self.tr_multiplier, self.length_max_discrete))
            self.length = min(self.length * self.tr_multiplier, self.length_max)
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.failcount = 0
            # Ditto for shrinking.
            self.length_discrete = int(self.length_discrete / self.tr_multiplier)
            self.length = max(self.length / self.tr_multiplier, self.length_min)
            print("Shrink", self.length, self.length_discrete)

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers, return_acq=False):
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                cat_dims=self.cat_dims, cont_dims=self.cont_dims,
                int_constrained_dims=self.int_constrained_dims,
                noise_variance=self.kwargs['noise_variance'] if 'noise_variance' in self.kwargs else None
            )
            # Save state dict
            hypers = gp.state_dict()

        from .localbo_utils import interleaved_search
        x_center = X[fX.argmin().item(), :][None, :]
        # Compute the trust region boundaries for the continuous variables
        weights = gp.covar_module.base_kernel.continuous_kern.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        # Note that the length supplied is the discrete length; here we need to call self.length which is for the
        # continuous variables
        lb = np.clip(x_center[0][self.cont_dims] - weights * self.length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center[0][self.cont_dims] + weights * self.length / 2.0, 0.0, 1.0)

        def thompson(n_cand=5000):
            """Thompson sampling"""
            # Generate n_cand of candidates for the discrete variables, in their trust region
            X_cand_cat = np.array([
                random_sample_within_discrete_tr_ordinal(x_center[0][self.cat_dims], length, self.config)
                for _ in range(n_cand)
            ])

            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(len(self.cont_dims), scramble=True, seed=seed)
            pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
            pert = lb + (ub - lb) * pert
            prob_perturb = min(20.0 / len(self.cont_dims), 1.0)
            mask = np.random.rand(n_cand, len(self.cont_dims)) <= prob_perturb
            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            mask[ind, np.random.randint(0, len(self.cont_dims) - 1, size=len(ind))] = 1

            X_cand_cont = x_center[0][self.cont_dims].copy() * np.ones((n_cand, len(self.cont_dims)))
            X_cand_cont[mask] = pert[mask]

            X_cand = np.hstack((X_cand_cat, X_cand_cont))

            # Generate n_cand candidates for the continuous variables, in their trust region
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                X_cand_torch = torch.tensor(X_cand, dtype=torch.float32)
                y_cand = gp.likelihood(gp(X_cand_torch)).sample(
                    torch.Size([self.batch_size])).t().cpu().detach().numpy()

            # Select the best candidates
            X_next = np.ones((self.batch_size, self.dim))
            y_next = np.ones((self.batch_size, 1))
            for i in range(self.batch_size):
                indbest = np.argmin(y_cand[:, i])
                X_next[i, :] = deepcopy(X_cand[indbest, :])
                y_next[i, :] = deepcopy(y_cand[indbest, i])
                y_cand[indbest, :] = np.inf
            return X_next, y_next

        def _ei(X, augmented=True):
            """Expected improvement (with option to enable augmented EI"""
            from torch.distributions import Normal
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            gauss = Normal(torch.zeros(1), torch.ones(1))
            # flip for minimization problems
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size), \
                 gpytorch.settings.fast_pred_var():
                preds = gp(X)
                mean, std = -preds.mean, preds.stddev
                # use in-fill criterion
                mu_star = -gp.likelihood(gp(torch.tensor(x_center[0].reshape(1, -1), dtype=torch.float32))).mean

            u = (mean - mu_star) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf
            if augmented:
                sigma_n = gp.likelihood.noise
                ei *= (1. - torch.sqrt(torch.tensor(sigma_n)) / torch.sqrt(sigma_n + std ** 2))
            return ei

        def _ucb(X, beta=5.):
            """Upper confidence bound"""
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            # Invoked when you supply X in one-hot representations
            # if X.shape[1] == self.dim and self.dim != self.true_dim:
            #     X = onehot2ordinal(X, self.cat_dims)
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size), \
                 gpytorch.settings.fast_pred_var():
                preds = gp.likelihood(gp(X))
            mean, std = preds.mean, preds.stddev
            return -(mean + beta * std)

        if self.acq in ['ei', 'ucb']:
            if self.batch_size == 1:
                if self.acq == 'ei':
                    X_next, acq_next = interleaved_search(x_center[0], _ei,
                                                          self.cat_dims, self.cont_dims,
                                                          self.config, ub, lb, length, 3, self.batch_size,
                                                          interval=1)
                else:
                    X_next, acq_next = interleaved_search(x_center[0], _ucb,
                                                          self.cat_dims, self.cont_dims,
                                                          self.config, ub, lb, length, 3, self.batch_size,
                                                          interval=1)
            else:
                # batch setting: for these, we use the fantasised points {x, y}
                X_next = torch.tensor([], dtype=torch.float32)
                acq_next = np.array([])
                for p in range(self.batch_size):
                    if self.acq == 'ei':
                        x_next, acq = interleaved_search(x_center[0], _ei,
                                                         self.cat_dims, self.cont_dims,
                                                         self.config, ub, lb, length, 3, 1, interval=1)
                    else:
                        x_next, acq = interleaved_search(x_center[0], _ucb,
                                                         self.cat_dims, self.cont_dims,
                                                         self.config, ub, lb, length, 3, 1, interval=1)
                    x_next = torch.tensor(x_next, dtype=torch.float32)
                    # The fantasy point is filled by the posterior mean of the Gaussian process.
                    y_next = gp(x_next).mean.detach()
                    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                        X_torch = torch.cat((X_torch, x_next), dim=0)
                        y_torch = torch.cat((y_torch, y_next), dim=0)
                        gp = train_gp(
                            train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps,
                            hypers=hypers,
                            kern=self.kernel_type,
                            cat_dims=self.cat_dims, cont_dims=self.cont_dims,
                            int_constrained_dims=self.int_constrained_dims,
                            noise_variance=self.kwargs['noise_variance'] if 'noise_variance' in self.kwargs else None
                        )
                    X_next = torch.cat((X_next, x_next), dim=0)
                    acq_next = np.hstack((acq_next, acq))

        elif self.acq == 'thompson':
            X_next, acq_next = thompson()
        else:
            raise ValueError('Unknown acquisition function choice %s' % self.acq)

        X_next = np.array(X_next)
        if return_acq:
            return X_next, acq_next
        return X_next
