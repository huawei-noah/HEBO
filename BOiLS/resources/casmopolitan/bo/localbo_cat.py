# 2021.11.10-Add support for ssk
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from copy import deepcopy
from typing import Optional

import gpytorch
import math
import numpy as np
import sys
import torch

from core.algos.bo.boils.utils import InputTransformation, SentenceBertInputTransform
from resources.casmopolitan.bo.localbo_utils import train_gp, random_sample_within_discrete_tr_ordinal


class CASMOPOLITANCat:
    """

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
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

    Data types that require special treatments
    cat_dims: list of lists. e.g. [[1, 2], [3, 4, 5]], which denotes that indices 1,2,3,4,5 are categorical, and [1, 2]
        belong to the same variable (a categorical variable with 2 possible values) and [3, 4, 5] belong to another,
        with 3 possible values.
    int_dims: list. [2, 3, 4]. Denotes the indices of the dimensions that are of integer types

    true_dim: The actual dimension of the problem. When there is no categorical variables, this value would be the same
        as the dimensionality inferred from the data. When there are categorical variable(s), due to the one-hot
        transformation. If not supplied, the dimension inferred from the data will be used.

    """

    def __init__(
            self,
            dim,
            n_init,
            max_evals,
            config,
            device: int,
            batch_size=1,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            dtype="float32",
            acq='thompson',
            kernel_type='transformed_overlap',
            standardise: bool = False,
            **kwargs
    ):

        # Very basic input checks
        # assert lb.ndim == 1 and ub.ndim == 1
        # assert len(lb) == len(ub)
        # assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert dtype == "float32" or dtype == "float64"
        if device is None or device < 0:
            device = torch.device("cpu")
        else:
            assert torch.cuda.is_available()
            device = torch.device(f"cuda:{device}")
            torch.cuda.set_device(device)
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        # Save function information
        self.dim = dim
        self.config = config
        self.kwargs = kwargs
        # self.lb = lb
        # self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        self.acq = acq
        self.kernel_type = kernel_type
        self.input_transformation: Optional[InputTransformation] = self.kwargs.get('input_transformation')
        if 's-bert' in self.kernel_type:  # use input warping based on sentence-BERT embedding
            assert isinstance(self.input_transformation, SentenceBertInputTransform), type(self.input_transformation)
            self.input_transformation.sbert_model.eval()
            self.input_transformation.sbert_model.to(device=device, dtype=self.dtype)

        self.standardise = standardise

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = kwargs['n_cand'] if 'n_cand' in kwargs.keys() else min(100 * self.dim, 5000)
        self.tr_multiplier = kwargs['multiplier'] if 'multiplier' in kwargs.keys() else 1.5
        self.failtol = kwargs['failtol'] if 'failtol' in kwargs.keys() else 40
        self.succtol = kwargs['succtol'] if 'succtol' in kwargs.keys() else 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = kwargs['length_min'] if 'length_min' in kwargs.keys() else 0.5 ** 7
        self.length_max = kwargs['length_max'] if 'length_max' in kwargs.keys() else 1.6
        self.length_init = kwargs['length_init'] if 'length_init' in kwargs.keys() else 0.8

        # Trust region sizes (in terms of Hamming distance) of the discrete variables.
        self.length_min_discrete = kwargs['length_min_discrete'] if 'length_min_discrete' in kwargs.keys() else 1
        self.length_max_discrete = kwargs['length_max_discrete'] if 'length_max_discrete' in kwargs.keys() else 30
        self.length_init_discrete = kwargs['length_init_discrete'] if 'length_init_discrete' in kwargs.keys() else 20

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.embeddings = None
        if self.kwargs.get('input_transformation') is not None:
            self.embeddings = np.zeros((0, self.input_transformation.embed_dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda if kernel_type != 'ssk' else 0
        self.device = device
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()

    def _restart(self):
        self._X = []
        self._embeddings = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_discrete = self.length_init_discrete

    def adjust_tr_length(self, fX_next):
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([self.tr_multiplier * self.length, self.length_max])
            # For the Hamming distance-bounded trust region, we additively (instead of multiplicatively) adjust.
            self.length_discrete = int(min(self.length_discrete * self.tr_multiplier, self.length_max_discrete))
            # self.length = min(self.length * 1.5, self.length_max)
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            # self.length = max([self.length_min, self.length / 2.0])
            self.failcount = 0
            # Ditto for shrinking.
            self.length_discrete = int(self.length_discrete / self.tr_multiplier)
            # self.length = max(self.length / 1.5, self.length_min)
            print("Shrink", self.length, self.length_discrete)

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers,
                                      embedding_bounds=None,
                                      return_acq=False):
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            if self.standardise:
                y_torch = (y_torch - y_torch.mean()) / (y_torch.std() + 1e-8)

            if 's-bert' in self.kernel_type:
                assert self.input_transformation is not None
                assert embedding_bounds is not None
                embedding_bounds = embedding_bounds.to(device=device, dtype=dtype)
            if self.kernel_type == 'ssk':
                assert self.kwargs['alphabet_size'] is not None

            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                noise_variance=self.kwargs['noise_variance'] if
                'noise_variance' in self.kwargs else None,
                input_transformation=self.input_transformation,
                embedding_bounds=embedding_bounds,
                alphabet_size=self.kwargs['alphabet_size']
            )
            # Save state dict
            hypers = gp.state_dict()
        # Standardize function values.
        # mu, sigma = np.median(fX), fX.std()
        # sigma = 1.0 if sigma < 1e-6 else sigma
        # fX = (deepcopy(fX) - mu) / sigma

        from .localbo_utils import local_search
        x_center = X[fX.argmin().item(), :][None, :]

        def thompson(n_cand=5000):
            """Thompson sampling"""
            # Generate n_cand of candidates
            X_cand = np.array([
                random_sample_within_discrete_tr_ordinal(x_center[0], length, self.config)
                for _ in range(n_cand)
            ])
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                X_cand_torch = torch.tensor(X_cand, dtype=torch.float32)
                y_cand = gp.likelihood(gp(X_cand_torch)).sample(
                    torch.Size([self.batch_size])).t().cpu().detach().numpy()
            # Revert the normalization process
            # y_cand = mu + sigma * y_cand

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
                X = torch.tensor(X, dtype=self.dtype, device=self.device)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            gauss = Normal(torch.zeros(1).to(X), torch.ones(1).to(X))
            # flip for minimization problems
            preds = gp(X)
            mean, std = -preds.mean, preds.stddev
            # use in-fill criterion
            mu_star = -gp.likelihood(gp(torch.tensor(x_center[0].reshape(1, -1)).to(X))).mean

            u = (mean - mu_star) / std
            try:
                ucdf = gauss.cdf(u)
            except ValueError as e:
                raise ValueError(f'\t- {u}\n\t- {mean}\n\t- {mu_star}\n\t- {std}') from e

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

            preds = gp.likelihood(gp(X))
            mean, std = preds.mean, preds.stddev
            return -(mean + beta * std)

        if self.acq in ['ei', 'ucb']:
            if self.batch_size == 1:
                # Sequential setting
                if self.acq == 'ei':
                    X_next, acq_next = local_search(x_center[0], _ei, self.config, length, 3, self.batch_size)
                else:
                    X_next, acq_next = local_search(x_center[0], _ucb, self.config, length, 3, self.batch_size)

            else:
                # batch setting: for these, we use the fantasised points {x, y}
                X_next = torch.tensor([], dtype=torch.float32)
                acq_next = np.array([])
                for p in range(self.batch_size):
                    if self.acq == 'ei':
                        x_next, acq = local_search(x_center[0], _ei, self.config, length, 3, 1)
                    else:
                        x_next, acq = local_search(x_center[0], _ucb, self.config, length, 3, 1)
                    x_next = torch.tensor(x_next, dtype=torch.float32)
                    if embedding_bounds is not None:
                        # update embedding_bounds
                        embedding_next = self.input_transformation(x_next)
                        assert embedding_next.shape == (1, embedding_bounds.shape[1]), \
                            (embedding_next.shape, embedding_bounds.shape)
                        embedding_bounds[0] = torch.minimum(embedding_bounds[0], embedding_next[0].to(embedding_bounds))
                        embedding_bounds[1] = torch.maximum(embedding_bounds[1], embedding_next[0].to(embedding_bounds))
                    # The fantasy point is filled by the posterior mean of the Gaussian process.
                    y_next = gp(x_next).mean.detach()
                    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                        X_torch = torch.cat((X_torch, x_next), dim=0)
                        y_torch = torch.cat((y_torch, y_next), dim=0)
                        if 's-bert' in self.kernel_type:
                            assert 'input_transformation' in self.kwargs

                        gp = train_gp(
                            train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps,
                            kern=self.kernel_type,
                            hypers=hypers,
                            noise_variance=self.kwargs['noise_variance'] if
                            'noise_variance' in self.kwargs else None,
                            input_transformation=self.input_transformation,
                            embedding_bounds=embedding_bounds,
                            alphabet_size=self.kwargs['alphabet_size']
                        )
                    X_next = torch.cat((X_next, x_next), dim=0)
                    acq_next = np.hstack((acq_next, acq))

        elif self.acq == 'thompson':
            X_next, acq_next = thompson()
        else:
            raise ValueError('Unknown acquisition function choice %s' % self.acq)

        # Remove the torch tensors
        del X_torch, y_torch
        X_next = np.array(X_next)
        if return_acq:
            return X_next, acq_next
        return X_next
