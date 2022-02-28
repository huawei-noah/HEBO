# 2021.11.10-Add support for ssk
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from copy import deepcopy
from typing import Optional, List, Union

import numpy as np
import torch
import scipy.stats as ss
from gpytorch.utils.errors import NotPSDError, NanError

from core.algos.bo.boils.utils import InputTransformation
from resources.casmopolitan.bo.localbo_cat import CASMOPOLITANCat
from resources.casmopolitan.bo.localbo_utils import from_unit_cube, latin_hypercube, onehot2ordinal, \
    random_sample_within_discrete_tr_ordinal
from utils.utils_misc import log


def rank_standardise(y: Union[np.ndarray, List[float]]):
    y = np.nan_to_num(np.asarray(y))  # Get rid of inf values
    if y.ndim > 1:
        raise ValueError(f"Vector to standardise should be of ndim 1, got: {y.ndim}")
    # compute array of ranks: ranks[i] = rank of y[i] in y
    val, inverse_ind, count = np.unique(y, return_counts=True, return_inverse=True)
    ranks = np.cumsum(count)[inverse_ind]
    return ss.norm.ppf(ranks / (y.shape[0] + 1))  # ppf of quantiles


class Optimizer:

    def __init__(self, config,
                 n_init: int = None,
                 wrap_discrete: bool = True,
                 guided_restart: bool = True,
                 standardise: bool = False,
                 **kwargs):
        """
        Args:
            config: list. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
                being 2, 3, 4, and 5 respectively.
            guided_restart: whether to fit an auxiliary GP over the best points encountered in all previous restarts, and
                sample the points with maximum variance for the next restart.
            global_bo: whether to use the global version of the discrete GP without local modelling
            standardise: whether to standardise the output to fit the GP
        """

        # Maps the input order.
        self.config = config.astype(int)
        self.true_dim = len(config)
        self.kwargs = kwargs
        self.input_transformation: Optional[InputTransformation] = self.kwargs.get('input_transformation')
        if 's-bert' in self.kwargs['kernel_type']:  # use input warping based on sentence-BERT embedding
            assert self.input_transformation is not None
        if self.kwargs['kernel_type'] == 'ssk':
            assert 'alphabet_size' is not None and self.kwargs['alphabet_size'] is not None
        # Number of one hot dimensions
        self.n_onehot = int(np.sum(config))
        # One-hot bounds
        self.lb = np.zeros(self.n_onehot)
        self.ub = np.ones(self.n_onehot)
        self.dim = len(self.lb)
        # True dim is simply th`e number of parameters (do not care about one-hot encoding etc).
        self.history = []
        self.max_evals = int(1e9)
        self.batch_size = None
        self.wrap_discrete = wrap_discrete
        self.cat_dims = self.get_dim_info(config)

        self.casmopolitan = CASMOPOLITANCat(
            dim=self.true_dim,
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,
            max_evals=self.max_evals,
            batch_size=1,
            verbose=True,
            config=self.config,
            standardise=standardise,
            **kwargs
        )

        # Our modification: define an auxiliary GP
        self.guided_restart = guided_restart
        # keep track of the best X and fX in each restart
        self.best_X_each_restart, self.best_fX_each_restart, self.best_X_embedding_each_restart = None, None, None
        self.auxiliary_gp = None

    def restart(self):
        from resources.casmopolitan.bo.localbo_utils import train_gp

        if self.guided_restart and len(self.casmopolitan._fX):

            best_idx = self.casmopolitan._fX.argmin()
            # Obtain the best X and fX within each restart (bo._fX and bo._X get erased at each restart,
            # but bo.X and bo.fX always store the full history
            if self.best_fX_each_restart is None:
                self.best_fX_each_restart = deepcopy(self.casmopolitan._fX[best_idx])
                self.best_X_each_restart = deepcopy(self.casmopolitan._X[best_idx])
                if self.input_transformation is not None:
                    self.best_X_embedding_each_restart = deepcopy(self.casmopolitan._embeddings[best_idx])
            else:
                self.best_fX_each_restart = np.vstack(
                    (self.best_fX_each_restart, deepcopy(self.casmopolitan._fX[best_idx])))
                self.best_X_each_restart = np.vstack(
                    (self.best_X_each_restart, deepcopy(self.casmopolitan._X[best_idx])))
                if self.input_transformation is not None:
                    self.best_X_embedding_each_restart = np.vstack(
                        (self.best_X_embedding_each_restart, deepcopy(self.casmopolitan._embeddings[best_idx])))

            X_tr_torch = torch.tensor(self.best_X_each_restart, dtype=torch.float32).reshape(-1, self.true_dim)
            fX_tr_torch = torch.tensor(self.best_fX_each_restart, dtype=torch.float32).view(-1)
            if self.input_transformation is not None:
                X_embedding_tr_torch = torch.tensor(self.best_X_embedding_each_restart, dtype=torch.float32).reshape(-1,
                                                                                                                     self.true_dim)
            # Train the auxiliary
            self.auxiliary_gp = train_gp(X_tr_torch, fX_tr_torch, False, 400, )
            # Generate random points in a Thompson-style sampling
            X_init = latin_hypercube(self.casmopolitan.n_cand, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            if self.wrap_discrete:
                X_init = self.warp_discrete(X_init, )
            X_init = onehot2ordinal(X_init, self.cat_dims)
            with torch.no_grad():
                self.auxiliary_gp.eval()
                X_init_torch = torch.tensor(X_init, dtype=torch.float32)
                # LCB-sampling
                y_cand_mean, y_cand_var = self.auxiliary_gp(
                    X_init_torch).mean.cpu().detach().numpy(), self.auxiliary_gp(
                    X_init_torch).variance.cpu().detach().numpy()
                y_cand = y_cand_mean - 1.96 * np.sqrt(y_cand_var)

            self.X_init = np.ones((self.casmopolitan.n_init, self.true_dim))
            indbest = np.argmin(y_cand)
            # The initial trust region centre for the new restart
            centre = deepcopy(X_init[indbest, :])
            # The centre is the first point to be evaluated
            self.X_init[0, :] = deepcopy(centre)
            for i in range(1, self.casmopolitan.n_init):
                # Randomly sample within the initial trust region length around the centre
                self.X_init[i, :] = deepcopy(
                    random_sample_within_discrete_tr_ordinal(centre, self.casmopolitan.length_init_discrete,
                                                             self.config))
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            del X_tr_torch, fX_tr_torch, X_init_torch

        else:
            # If guided restart is not enabled, simply sample a number of points equal to the number of evaluated
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            self.X_init = from_unit_cube(latin_hypercube(self.casmopolitan.n_init, self.dim), self.lb, self.ub)
            if self.wrap_discrete:
                self.X_init = self.warp_discrete(self.X_init, )
            self.X_init = onehot2ordinal(self.X_init, self.cat_dims)
        if self.input_transformation is not None:
            self.casmopolitan._embeddings = np.zeros((0, self.input_transformation.embed_dim))

    def suggest(self, n_suggestions: int):
        if self.batch_size is None:
            self.casmopolitan.batch_size = n_suggestions
            self.batch_size = n_suggestions
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])
            self.restart()

        X_suggest = np.zeros((n_suggestions, self.true_dim))

        # Choose among init samples
        n_initial = min(self.X_init.shape[0], n_suggestions)
        if n_initial > 0:
            X_suggest[:n_initial] = deepcopy(self.X_init[:n_initial, :])
            self.X_init = self.X_init[n_initial:, :]

        # Acquire the remaining points using acquisition function optimisation
        n_acq_func_opt_pts = n_suggestions - n_initial
        if n_acq_func_opt_pts > 0:
            if len(self.casmopolitan._X) > 0:  # If not enough point to fit the model, get random samples
                X = deepcopy(self.casmopolitan._X)
                fX = rank_standardise(deepcopy(self.casmopolitan._fX).ravel())
                embedding_bounds = None
                if self.input_transformation is not None:
                    embedding_bounds = torch.zeros(2, self.input_transformation.embed_dim)
                    embedding_bounds[0] = torch.from_numpy(self.casmopolitan._embeddings.min(0))
                    embedding_bounds[1] = torch.from_numpy(self.casmopolitan._embeddings.max(0))
                try:
                    X_suggest[-n_acq_func_opt_pts:, :] = self.casmopolitan._create_and_select_candidates(
                        X=X,
                        fX=fX,
                        length=self.casmopolitan.length_discrete,
                        n_training_steps=500,
                        hypers={},
                        embedding_bounds=embedding_bounds,
                    )[-n_acq_func_opt_pts:, :]
                except (ValueError, NotPSDError, NanError):
                    assert self.casmopolitan.kernel_type == 'ssk'  # can get instability with this kernel sometimes
                    self.casmopolitan.kernel_type = 'transformed_overlap'
                    log("Failed acquisition with ssk, try with original Casmopolitan kernel", header="CASMOPOLITAN")
                    X_suggest[-n_acq_func_opt_pts:, :] = self.casmopolitan._create_and_select_candidates(
                        X,
                        fX,
                        length=self.casmopolitan.length_discrete,
                        n_training_steps=500,
                        embedding_bounds=embedding_bounds,
                        hypers={},
                    )[-n_acq_func_opt_pts:, :]
                    self.casmopolitan.kernel_type = 'ssk'
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return X_suggest

    def observe(self, X: np.ndarray, y: np.ndarray):
        """ Observe the blackbox values at a set of points

        Args:
            X : array of newly observed points
            y : array of blackbox values
        """
        # XX = torch.cat([ordinal2onehot(x, self.n_categories) for x in X]).reshape(len(X), -1)
        y = deepcopy(np.array(y)[:, None])
        # if self.wrap_discrete:
        #     X = self.warp_discrete(X, )
        X = deepcopy(X)
        assert X.shape[0] == y.shape[0], (X, y.shape)

        if self.casmopolitan._fX.shape[0] >= self.casmopolitan.n_init:
            self.casmopolitan.adjust_tr_length(y)

        self.casmopolitan._X = np.vstack((self.casmopolitan._X, X))
        self.casmopolitan._fX = np.vstack((self.casmopolitan._fX, y.reshape(-1, 1)))
        if self.input_transformation is not None:
            new_embeds = deepcopy(self.input_transformation(X))
            self.casmopolitan._embeddings = np.vstack((self.casmopolitan._embeddings, new_embeds))
            self.casmopolitan.embeddings = np.vstack((self.casmopolitan.embeddings, new_embeds))
        self.casmopolitan.fX = np.vstack((self.casmopolitan.fX, y.reshape(-1, 1)))
        self.casmopolitan.X = np.vstack((self.casmopolitan.X, X))
        self.casmopolitan.n_evals += self.batch_size

        if self.casmopolitan.length <= self.casmopolitan.length_min or self.casmopolitan.length_discrete <= self.casmopolitan.length_min_discrete:
            self.restart()

    def warp_discrete(self, X, ):

        X_ = np.copy(X)
        # Process the integer dimensions
        if self.cat_dims is not None:
            for categorical_groups in self.cat_dims:
                max_col = np.argmax(X[:, categorical_groups], axis=1)
                X_[:, categorical_groups] = 0
                for idx, g in enumerate(max_col):
                    X_[idx, categorical_groups[g]] = 1
        return X_

    def get_dim_info(self, n_categories):
        dim_info = []
        offset = 0
        for i, cat in enumerate(n_categories):
            dim_info.append(list(range(offset, offset + cat)))
            offset += cat
        return dim_info
