import os
import sys
from copy import deepcopy
from typing import Optional, Any, Union

import gpytorch
import math
import numpy as np
import torch
from sklearn.preprocessing import power_transform

from bo.gp import train_gp
from bo.localbo_utils import random_sample_within_discrete_tr_ordinal, ACQ_FUNCTIONS, SEARCH_STRATS, \
    local_table_emmbedding_search
from bo.utils import update_table_of_candidates
from utilities.misc_utils import filter_kwargs


def hebo_transform(y: np.ndarray) -> np.ndarray:
    try:
        if y.min() <= 0:
            y_transform = power_transform(y / y.std(), method='yeo-johnson')
        else:
            y_transform = power_transform(y / y.std(), method='box-cox')
            if y_transform.std() < 0.5:
                y_transform = power_transform(y / y.std(), method='yeo-johnson')
        if y_transform.std() < 0.5:
            raise RuntimeError('Power transformation failed')
        return y_transform
    except RuntimeError:
        return (y - y.mean()) / (y.std() + 1e-8)


class CASMOPOLITANCat:
    """

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
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
            self, dim: int, n_init: int, config: np.ndarray, batch_size: int = 1,
            verbose: bool = True, use_ard: bool = True, cdr_constraints: bool = False,
            max_cholesky_size: int = 2000, n_training_steps: int = 50, min_cuda: int = 1024,
            normalise: bool = False, device: str = "cpu", dtype: str = "float32",
            acq: ACQ_FUNCTIONS = 'thompson', kernel_type: str = 'transformed_overlap', seed: int = 0,
            search_strategy: SEARCH_STRATS = 'local', **kwargs
    ) -> None:
        # Very basic input checks
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
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
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device(device) if "cuda" in device else torch.device("cpu")

        # self.lb = lb
        # self.ub = ub
        # Settings
        self.n_init = n_init
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
        self.antigen = None
        if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
            from transformers import BertModel, BertTokenizer

            self.BERT_model = BertModel.from_pretrained(pretrained_model_name_or_path=kwargs['BERT_model_path']).to(
                device=self.device, dtype=self.dtype)
            self.BERT_tokeniser = BertTokenizer.from_pretrained(self.kwargs['BERT_tokeniser_path'], do_lower_case=False)
            self.BERT_batchsize = self.kwargs['BERT_batchsize']
            if self.kernel_type in ['rbf-pca-BERT', 'cosine-pca-BERT']:
                self.antigen = self.kwargs['antigen']
        else:
            self.BERT_model, self.BERT_tokeniser, self.BERT_batchsize = None, None, None

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
        self.length_max = kwargs['length_max'] if 'length_max' in kwargs.keys() else 1.
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
        self.x = np.zeros((0, self.dim))
        self.fx = np.zeros((0, 1))

        self.min_cuda = min_cuda if kernel_type != 'ssk' else 0
        # Device and dtype for GPyTorch
        # self.min_cuda = min_cuda
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self.succcount = None
        self.failcount = None
        self.length_discrete = None
        self.length = None
        self.tr_x = None
        self.tr_fx = None
        self.restart()

    def restart(self) -> None:
        self.tr_x = np.zeros((0, self.dim))
        self.tr_fx = np.zeros((0, 1))
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_discrete = self.length_init_discrete

    def adjust_length(self, fx_next: np.ndarray) -> None:
        if np.min(fx_next) <= np.min(self.tr_fx) - 1e-3 * math.fabs(np.min(self.tr_fx)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            # For the Hamming distance-bounded trust region, we additively (instead of multiplicatively) adjust.
            self.length_discrete = int(min(self.length_discrete * self.tr_multiplier, self.length_max_discrete))
            self.length = min(self.length * self.tr_multiplier, self.length_max)
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.failcount = 0
            # Ditto for shrinking.
            self.length_discrete = int(self.length_discrete / self.tr_multiplier)
            self.length = max(self.length / self.tr_multiplier, self.length_min)

    def create_and_select_candidates(
            self, x: np.array, fx: np.array, n_training_steps: int, hypers: dict[str, Any],
            table_of_candidates: Optional[np.ndarray], table_of_candidate_embeddings: Optional[np.ndarray],
            embedding_from_array_dict: Optional[dict[str, np.ndarray]],
    ) -> np.ndarray:
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if self.search_strategy == 'glocal':
            fx = hebo_transform(y=fx)
        else:
            fx = (fx - fx.mean()) / (fx.std() + 1e-8)

        if len(x) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if table_of_candidate_embeddings is not None:
            x_for_gp = np.array([embedding_from_array_dict[str(xx.astype(int))] for xx in x])
        else:
            x_for_gp = x

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            x_torch = torch.tensor(x_for_gp).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fx).to(device=device, dtype=dtype)
            if self.kernel_type == 'ssk':
                assert self.kwargs['alphabet_size'] is not None
            gp = train_gp(
                train_x=x_torch, train_y=y_torch, use_ard=self.use_ard,
                num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                noise_variance=self.kwargs['noise_variance'] if 'noise_variance' in self.kwargs else None,
                alphabet_size=self.kwargs['alphabet_size'],
                BERT_model=self.BERT_model,
                BERT_tokeniser=self.BERT_tokeniser,
                BERT_batchsize=self.BERT_batchsize,
                antigen=self.antigen,
                search_strategy=self.search_strategy
            )
            # Save state dict
            hypers = gp.state_dict()

            if table_of_candidate_embeddings is not None:
                weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
                weights = weights / weights.mean()  # This will make the next line more stable
                weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
                per_dim_max_dist = np.clip(weights * self.length / 2.0, 0, 1)
            else:
                per_dim_max_dist = None

        # Standardize function values
        from bo.localbo_utils import local_search, glocal_search, local_table_search

        if table_of_candidates is not None:
            if table_of_candidate_embeddings is None:
                search = local_table_search
            else:
                search = local_table_emmbedding_search
        elif self.search_strategy == 'glocal':
            search = glocal_search
        else:
            search = local_search

        center_ind = fx.argmin().item()
        x_center_for_gp = x_for_gp[center_ind, :][None, :]
        x_center_as_ind = x[center_ind, :][None, :]

        def thompson(n_cand: int = 5000) -> tuple[np.ndarray, np.ndarray]:
            """Thompson sampling"""
            # Generate n_cand of candidates
            if table_of_candidate_embeddings is not None:
                raise NotImplementedError()
            x_cand = np.array([
                random_sample_within_discrete_tr_ordinal(x_center_for_gp[0], self.length_discrete, self.config)
                for _ in range(n_cand)
            ])
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                x_cand_torch = torch.tensor(x_cand, dtype=torch.float32)
                y_cand = gp.likelihood(gp(x_cand_torch)).sample(
                    torch.Size([self.batch_size])).t().cpu().detach().numpy()
            # Revert the normalization process
            # y_cand = mu + sigma * y_cand

            # Select the best candidates
            x_next_ = np.ones((self.batch_size, self.dim))
            y_next_ = np.ones((self.batch_size, 1))
            for i in range(self.batch_size):
                indbest = np.argmin(y_cand[:, i])
                x_next_[i, :] = deepcopy(x_cand[indbest, :])
                y_next_[i, :] = deepcopy(y_cand[indbest, i])
                y_cand[indbest, :] = np.inf
            return x_next_, y_next_

        def _ei(x_: Union[np.ndarray, torch.tensor], augmented: bool = True) -> tuple[np.ndarray, np.ndarray]:
            """ Expected improvement (with option to enable augmented EI """
            from torch.distributions import Normal
            if not isinstance(x_, torch.Tensor):
                x_ = torch.tensor(x_, dtype=torch.float32)
            if x_.dim() == 1:
                x_ = x_.reshape(1, -1)
            gauss = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
            # flip for minimization problems
            if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
                if table_of_candidate_embeddings is not None:
                    raise RuntimeError()
                from bo.utils import BERTFeatures
                bert_feats = BERTFeatures(self.BERT_model, self.BERT_tokeniser)
                x_reprsn = bert_feats.compute_features(x_.to(device))
                x_center_reprsn = bert_feats.compute_features(torch.tensor(x_center_for_gp[0].reshape(1, -1)))
                if self.kernel_type in ['rbf-pca-BERT', 'cosine-pca-BERT']:
                    from joblib import load
                    pca_ = load(f"/nfs/aiml/asif/CDRdata/pca/{self.antigen}_pca.joblib")
                    scaler_ = load(f"/nfs/aiml/asif/CDRdata/pca/{self.antigen}_scaler.joblib")
                    x_reprsn = torch.from_numpy(pca_.transform(scaler.transform(x_reprsn.cpu().numpy())))
                    x_center_reprsn = torch.from_numpy(pca_.transform(scaler_.transform(x_center_reprsn.cpu().numpy())))
                del bert_feats
                preds = gp(x_reprsn.to(device))
                # use in-fill criterion
                mu_star = -gp.likelihood(gp(x_center_reprsn.to(device))).mean
            else:
                preds = gp(x_.to(device))
                # use in-fill criterion
                posterior = gp(torch.tensor(x_center_for_gp[0].reshape(1, -1), dtype=torch.float32).to(device))
                mu_star = -gp.likelihood(posterior).mean
            mean, std = -preds.mean, preds.stddev
            u = (mean - mu_star) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf
            if augmented:
                sigma_n = gp.likelihood.noise
                ei *= (1. - torch.sqrt(sigma_n.clone().detach()) / torch.sqrt(sigma_n + std ** 2))
            return ei

        def _ucb(x_: np.ndarray, beta: float = 5.) -> torch.tensor:
            """Upper confidence bound"""
            if not isinstance(x_, torch.Tensor):
                x_ = torch.tensor(x_, dtype=torch.float32)
            if x_.dim() == 1:
                x_ = x_.reshape(1, -1)
            # Invoked when you supply X in one-hot representations

            preds = gp.likelihood(gp(x_))
            mean, std = preds.mean, preds.stddev
            return mean + beta * std

        if self.acq in ['ei', 'ucb']:
            if self.batch_size == 1:
                # Sequential setting
                if self.acq == 'ei':
                    f_acq = _ei
                else:
                    f_acq = _ucb
                search_kwargs = dict(
                    x_center=x_center_as_ind[0], f=f_acq, config=self.config,
                    max_hamming_dist=self.length_discrete,
                    max_cont_dist=self.length,
                    n_restart=3, batch_size=self.batch_size,
                    cdr_constraints=self.cdr_constraints, seed=self.seed, dtype=self.dtype,
                    device=device, table_of_candidates=table_of_candidates,
                    table_of_candidate_embeddings=table_of_candidate_embeddings,
                    embedding_from_array_dict=embedding_from_array_dict,
                    per_dim_max_dist=per_dim_max_dist
                )
                x_next, acq_next = search(**filter_kwargs(function=search, **search_kwargs))
            else:
                # batch setting: for these, we use the fantasised points {x, y}
                x_next = np.zeros((0, x_center_as_ind.shape[1]))
                acq_next = np.array([])
                for _ in range(self.batch_size):
                    if self.acq == 'ei':
                        f_acq = _ei
                    else:
                        f_acq = _ucb
                    search_kwargs = dict(
                        x_center=x_center_as_ind[0], f=f_acq, config=self.config, max_hamming_dist=self.length_discrete,
                        max_cont_dist=self.length, n_restart=3, batch_size=1, cdr_constraints=self.cdr_constraints,
                        seed=self.seed, dtype=self.dtype, device=device, table_of_candidates=table_of_candidates,
                        table_of_candidate_embeddings=table_of_candidate_embeddings,
                        embedding_from_array_dict=embedding_from_array_dict, per_dim_max_dist=per_dim_max_dist
                    )
                    x_next_step, acq = search(**filter_kwargs(function=search, **search_kwargs))
                    if table_of_candidates is not None:
                        table_of_candidates, table_of_candidate_embeddings = update_table_of_candidates(
                            original_table=table_of_candidates,
                            table_of_candidate_embeddings=table_of_candidate_embeddings,
                            observed_candidates=x_next_step,
                            check_candidates_in_table=True
                        )

                    if table_of_candidate_embeddings is None:
                        x_next_step_gp_aux = x_next_step
                    else:
                        x_next_step_gp_aux = embedding_from_array_dict[str(x_next_step.flatten().astype(int))].reshape(
                            1, -1)
                    x_next_step_gp = torch.tensor(x_next_step_gp_aux, dtype=torch.float32, device=device)
                    # The fantasy point is filled by the posterior mean of the Gaussian process.
                    if self.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
                        from bo.utils import BERTFeatures
                        bert = BERTFeatures(self.BERT_model, self.BERT_tokeniser)
                        x_next_reprsn = bert.compute_features(x_next_step_gp)
                        # x_next_reprsn = rearrange(x_next_reprsn, 'b l d -> b (l d)')
                        if self.kernel_type in ['rbf-pca-BERT', 'cosine-pca-BERT']:
                            from joblib import load
                            pca = load(f"{self.antigen}_pca.joblib")
                            scaler = load(f"{self.antigen}_scaler.joblib")
                            x_next_reprsn = torch.from_numpy(
                                pca.transform(scaler.transform(x_next_reprsn.cpu().numpy())))
                        del bert
                        y_next = gp(x_next_reprsn).mean.detach()
                    else:
                        y_next = gp(x_next_step_gp).mean.detach()
                    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                        x_torch = torch.cat((x_torch, x_next_step_gp), dim=0).to(device=device)
                        y_torch = torch.cat((y_torch, y_next), dim=0).to(device=device)
                        gp = train_gp(
                            train_x=x_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps,
                            kern=self.kernel_type,
                            hypers=hypers,
                            noise_variance=self.kwargs['noise_variance'] if 'noise_variance' in self.kwargs else None,
                            alphabet_size=self.kwargs['alphabet_size'],
                            BERT_model=self.BERT_model,
                            BERT_tokeniser=self.BERT_tokeniser,
                            BERT_batchsize=self.BERT_batchsize,
                            antigen=self.antigen,
                            search_strategy=self.search_strategy
                        )
                    x_next = np.concatenate([x_next, x_next_step])
                    acq_next = np.hstack((acq_next, acq))

        elif self.acq == 'thompson':
            x_next, acq_next = thompson()
        else:
            raise ValueError('Unknown acquisition function choice %s' % self.acq)

        del x_torch, y_torch
        return x_next
