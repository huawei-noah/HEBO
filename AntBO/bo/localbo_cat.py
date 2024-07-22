import os
import sys
from copy import deepcopy
from typing import Optional

import gpytorch
import math
import numpy as np
import torch
from sklearn.preprocessing import power_transform

from bo.gp import train_gp
from bo.localbo_utils import random_sample_within_discrete_tr_ordinal
from bo.utils import update_table_of_candidates


def hebo_transform(X):
    try:
        if X.min() <= 0:
            y = power_transform(X / X.std(), method='yeo-johnson')
        else:
            y = power_transform(X / X.std(), method='box-cox')
            if y.std() < 0.5:
                y = power_transform(X / X.std(), method='yeo-johnson')
        if y.std() < 0.5:
            raise RuntimeError('Power transformation failed')
        return y
    except:
        return (X - X.mean()) / (X.std() + 1e-8)


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
            batch_size=1,
            verbose=True,
            use_ard=True,
            cdr_constraints=False,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            normalise=False,
            device="cpu",
            dtype="float32",
            acq='thompson',
            kernel_type='transformed_overlap',
            seed=0,
            search_strategy='local',
            **kwargs
    ):

        # Very basic input checks
        # assert lb.ndim == 1 and ub.ndim == 1
        # assert len(lb) == len(ub)
        # assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        # assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        self.seed = seed
        self.search_strategy = search_strategy
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
        self.cdr_constraints = cdr_constraints
        self.normalise = normalise
        if self.search_strategy != 'local':
            self.kwargs['noise_variance'] = None
        self.acq = acq
        self.kernel_type = kernel_type
        if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
            self.BERT_model = self.kwargs['BERT_model']
            self.BERT_tokeniser = self.kwargs['BERT_tokeniser']
            self.BERT_batchsize = self.kwargs['BERT_batchsize']
            self.antigen = self.kwargs['antigen']
        else:
            self.BERT_model, self.BERT_tokeniser, self.BERT_batchsize = None, None, None
            self.antigen = None

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = kwargs['n_cand'] if 'n_cand' in kwargs.keys() else min(100 * self.dim, 5000)
        self.tr_multiplier = kwargs['multiplier'] if 'multiplier' in kwargs.keys() else 1.5
        if os.getenv("ANTBO_DEBUG", False):
            failtol = 5
            succtol = 3
        else:
            succtol = kwargs['succtol'] if 'succtol' in kwargs.keys() else 3
            failtol = kwargs['failtol'] if 'failtol' in kwargs.keys() else 40
        self.succtol = succtol
        self.failtol = failtol
        self.n_evals = 0

        # Trust region sizes
        self.length_min = kwargs['length_min'] if 'length_min' in kwargs.keys() else 0.5 ** 7
        self.length_max = kwargs['length_max'] if 'length_max' in kwargs.keys() else 1.6
        self.length_init = kwargs['length_init'] if 'length_init' in kwargs.keys() else 0.8

        # Trust region sizes (in terms of Hamming distance) of the discrete variables.
        self.length_min_discrete = kwargs['length_min_discrete'] if 'length_min_discrete' in kwargs.keys() else 1
        if os.getenv("ANTBO_DEBUG", False):
            lmd = 10
        else:
            lmd = kwargs['length_max_discrete'] if 'length_max_discrete' in kwargs.keys() else min(30, self.dim)
        self.length_max_discrete = lmd
        if os.getenv("ANTBO_DEBUG", False):
            lid = lmd
        else:
            lid = kwargs['length_init_discrete'] if 'length_init_discrete' in kwargs.keys() else min(20, self.dim)
        self.length_init_discrete = lid

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        self.min_cuda = min_cuda if kernel_type != 'ssk' else 0
        # Device and dtype for GPyTorch
        # self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()

    def _restart(self):
        self._X = np.zeros((0, self.dim))
        self._fX = np.zeros((0, 1))
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_discrete = self.length_init_discrete

    def _adjust_length(self, fX_next):
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

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers,
                                      table_of_candidates: Optional[np.ndarray], return_acq=False):
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if self.search_strategy == 'glocal':
            fX = hebo_transform(fX)
        else:
            fX = (fX - fX.mean()) / (fX.std() + 1e-8)

        global device
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            if self.kernel_type == 'ssk':
                assert self.kwargs['alphabet_size'] is not None
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard,
                num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                noise_variance=self.kwargs['noise_variance'] if
                'noise_variance' in self.kwargs else None,
                alphabet_size=self.kwargs['alphabet_size'],
                BERT_model=self.BERT_model,
                BERT_tokeniser=self.BERT_tokeniser,
                BERT_batchsize=self.BERT_batchsize,
                antigen=self.antigen,
                search_strategy=self.search_strategy
            )
            # Save state dict
            hypers = gp.state_dict()
        # Standardize function values.
        # mu, sigma = np.median(fX), fX.std()
        # sigma = 1.0 if sigma < 1e-6 else sigma
        # fX = (deepcopy(fX) - mu) / sigma
        from bo.localbo_utils import local_search, glocal_search, local_table_search

        if table_of_candidates is not None:
            search = local_table_search
        elif self.search_strategy == 'glocal':
            search = glocal_search
        else:
            search = local_search

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
                X = torch.tensor(X, dtype=torch.float32)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            gauss = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
            # flip for minimization problems
            if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
                from bo.utils import BERTFeatures
                from einops import rearrange
                bert = BERTFeatures(self.BERT_model, self.BERT_tokeniser)
                x_reprsn = bert.compute_features(X)
                x_reprsn = rearrange(x_reprsn, 'b l d -> b (l d)')
                x_center_reprsn = bert.compute_features(torch.tensor(x_center[0].reshape(1, -1)))
                x_center_reprsn = rearrange(x_center_reprsn, 'b l d -> b (l d)')
                if self.kernel_type in ['rbf-pca-BERT', 'cosine-pca-BERT']:
                    from joblib import load
                    pca = load(f"/nfs/aiml/asif/CDRdata/pca/{self.antigen}_pca.joblib")
                    scaler = load(f"/nfs/aiml/asif/CDRdata/pca/{self.antigen}_scaler.joblib")
                    x_reprsn = torch.from_numpy(pca.transform(scaler.transform(x_reprsn.cpu().numpy())))
                    x_center_reprsn = torch.from_numpy(pca.transform(scaler.transform(x_center_reprsn.cpu().numpy())))
                del bert
                preds = gp(x_reprsn.to(device))
                # use in-fill criterion
                mu_star = -gp.likelihood(gp(x_center_reprsn.to(device))).mean
            else:
                preds = gp(X.to(device))
                # use in-fill criterion
                mu_star = -gp.likelihood(
                    gp(torch.tensor(x_center[0].reshape(1, -1), dtype=torch.float32).to(device))).mean
            mean, std = -preds.mean, preds.stddev
            u = (mean - mu_star) / std
            ucdf = gauss.cdf(u)
            # try:
            #     ucdf = gauss.cdf(u)
            # except ValueError as e:
            #     raise ValueError(f"\t- {u}\n\t- {mean}\n\t- {mu_star}\n\t- {std}") from e

            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf
            if augmented:
                sigma_n = gp.likelihood.noise
                ei *= (1. - torch.sqrt(sigma_n.clone().detach()) / torch.sqrt(sigma_n + std ** 2))

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
            return mean + beta * std

        if self.acq in ['ei', 'ucb']:
            if self.batch_size == 1:
                # Sequential setting
                if self.acq == 'ei':
                    X_next, acq_next = search(x_center=x_center[0], f=_ei, config=self.config, max_hamming_dist=length,
                                              n_restart=3, batch_size=self.batch_size,
                                              cdr_constraints=self.cdr_constraints, seed=self.seed, dtype=self.dtype,
                                              device=self.device, table_of_candidates=table_of_candidates)
                else:
                    X_next, acq_next = search(x_center=x_center[0], f=_ucb, config=self.config, max_hamming_dist=length,
                                              n_restart=3, batch_size=self.batch_size,
                                              cdr_constraints=self.cdr_constraints, seed=self.seed, dtype=self.dtype,
                                              device=self.device, table_of_candidates=table_of_candidates)
            else:
                # batch setting: for these, we use the fantasised points {x, y}
                X_next = torch.tensor([], dtype=torch.float32)
                acq_next = np.array([])
                for p in range(self.batch_size):
                    if self.acq == 'ei':
                        x_next, acq = search(x_center=x_center[0], f=_ei, config=self.config, max_hamming_dist=length,
                                             n_restart=3, batch_size=1, cdr_constraints=self.cdr_constraints,
                                             seed=self.seed, dtype=self.dtype, device=self.device,
                                             table_of_candidates=table_of_candidates)
                    else:
                        x_next, acq = search(x_center=x_center[0], f=_ucb, config=self.config, max_hamming_dist=length,
                                             n_restart=3, batch_size=1, cdr_constraints=self.cdr_constraints,
                                             seed=self.seed, dtype=self.dtype, device=self.device,
                                             table_of_candidates=table_of_candidates)

                    table_of_candidates = update_table_of_candidates(
                        original_table=table_of_candidates,
                        observed_candidates=x_next,
                        check_candidates_in_table=True
                    )

                    x_next = torch.tensor(x_next, dtype=torch.float32, device=self.device)
                    # The fantasy point is filled by the posterior mean of the Gaussian process.
                    if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
                        from bo.utils import BERTFeatures
                        from einops import rearrange
                        bert = BERTFeatures(self.BERT_model, self.BERT_tokeniser)
                        x_next_reprsn = bert.compute_features(x_next)
                        x_next_reprsn = rearrange(x_next_reprsn, 'b l d -> b (l d)')
                        if self.kernel_type in ['rbf-pca-BERT', 'cosine-pca-BERT']:
                            from joblib import load
                            pca = load(f"{self.antigen}_pca.joblib")
                            scaler = load(f"{self.antigen}_scaler.joblib")
                            x_next_reprsn = torch.from_numpy(
                                pca.transform(scaler.transform(x_next_reprsn.cpu().numpy())))
                        del bert
                        y_next = gp(x_next_reprsn).mean.detach()
                    else:
                        y_next = gp(x_next).mean.detach()
                    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                        X_torch = torch.cat((X_torch, x_next), dim=0).to(device=self.device)
                        y_torch = torch.cat((y_torch, y_next), dim=0).to(device=self.device)
                        gp = train_gp(
                            train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps,
                            kern=self.kernel_type,
                            hypers=hypers,
                            noise_variance=self.kwargs['noise_variance'] if
                            'noise_variance' in self.kwargs else None,
                            alphabet_size=self.kwargs['alphabet_size'],
                            BERT_model=self.BERT_model,
                            BERT_tokeniser=self.BERT_tokeniser,
                            BERT_batchsize=self.BERT_batchsize,
                            antigen=self.antigen,
                            search_strategy=self.search_strategy
                        )
                    X_next = torch.cat((X_next, x_next), dim=0)
                    acq_next = np.hstack((acq_next, acq))

        elif self.acq == 'thompson':
            X_next, acq_next = thompson()
        else:
            raise ValueError('Unknown acquisition function choice %s' % self.acq)

        del X_torch, y_torch
        X_next = np.array(X_next)
        if return_acq:
            return X_next, acq_next
        return X_next
