from copy import deepcopy
from itertools import groupby
import numpy as np
import scipy.stats as ss
from bo.localbo_cat import CASMOPOLITANCat
from bo.localbo_utils import from_unit_cube, latin_hypercube, to_unit_cube, ordinal2onehot, onehot2ordinal,\
    random_sample_within_discrete_tr_ordinal
from bo.localbo_utils import check_cdr_constraints
import torch
import logging
from gpytorch.utils.errors import NotPSDError, NanError
from utilities.constraint_utils import check_constraint_satisfaction_batch

COUNT_AA = 5

def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats

def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class Optimizer:

    def __init__(self,
                 config: np.ndarray,
                 min_cuda,
                 batch_size: int =1,
                 normalise: bool = False,
                 cdr_constraints: bool = False,
                 n_init: int = None,
                 wrap_discrete: bool = True,
                 guided_restart: bool = True,
                 **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        config: list. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
            being 2, 3, 4, and 5 respectively.
        guided_restart: whether to fit an auxiliary GP over the best points encountered in all previous restarts, and
            sample the points with maximum variance for the next restart.
        global_bo: whether to use the global version of the discrete GP without local modelling
        """

        # Maps the input order.
        self.config = config.astype(int)
        self.true_dim = len(config)
        self.kwargs = kwargs
        if self.kwargs['kernel_type'] == 'ssk':
            assert 'alphabet_size' != None and self.kwargs['alphabet_size'] != None
        # Number of one hot dimensions
        self.n_onehot = int(np.sum(config))
        # One-hot bounds
        self.lb = np.zeros(self.n_onehot)
        self.ub = np.ones(self.n_onehot)
        self.dim = len(self.lb)
        # True dim is simply th`e number of parameters (do not care about one-hot encoding etc).
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []
        self.wrap_discrete = wrap_discrete
        self.cat_dims = self.get_dim_info(config)
        self.cdr_constraints = cdr_constraints
        self.casmopolitan = CASMOPOLITANCat(
            min_cuda=min_cuda,
            dim=self.true_dim,
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,
            max_evals=self.max_evals,
            cdr_constraints = self.cdr_constraints,
            normalise = normalise,
            batch_size=1,  # We need to update this later
            verbose=False,
            config=self.config,
            **kwargs
        )

        # Our modification: define an auxiliary GP
        self.guided_restart = guided_restart
        # keep track of the best X and fX in each restart
        self.best_X_each_restart, self.best_fX_each_restart = None, None
        self.auxiliary_gp = None
        self.X_init, itern = [], 0

    def restart(self):
        from bo.gp import train_gp
        if self.guided_restart and len(self.casmopolitan._fX) and self.kwargs['search_strategy'] in ['local', 'batch_local']:
            best_idx = self.casmopolitan._fX.argmin()
            # Obtain the best X and fX within each restart (bo._fX and bo._X get erased at each restart,
            # but bo.X and bo.fX always store the full history
            if self.best_fX_each_restart is None:
                self.best_fX_each_restart = deepcopy(self.casmopolitan._fX[best_idx])
                self.best_X_each_restart = deepcopy(self.casmopolitan._X[best_idx])
            else:
                self.best_fX_each_restart = np.vstack((self.best_fX_each_restart, deepcopy(self.casmopolitan._fX[best_idx])))
                self.best_X_each_restart = np.vstack((self.best_X_each_restart, deepcopy(self.casmopolitan._X[best_idx])))

            X_tr_torch = torch.tensor(self.best_X_each_restart, dtype=torch.float32).reshape(-1, self.true_dim)
            fX_tr_torch = torch.tensor(self.best_fX_each_restart, dtype=torch.float32).view(-1)

            # Train the auxiliary
            self.auxiliary_gp = train_gp(X_tr_torch, fX_tr_torch, False, 300, )
            # Generate random points in a Thompson-style sampling
            X_init, itern = [], 0
            while(itern<self.casmopolitan.n_cand):
                X_init_itern = latin_hypercube(1, self.dim)
                X_init_itern = from_unit_cube(X_init_itern, self.lb, self.ub)
                if self.wrap_discrete:
                    X_init_itern = self.warp_discrete(X_init_itern, )
                X_init_itern = onehot2ordinal(X_init_itern, self.cat_dims)
                if self.cdr_constraints:
                    if not check_cdr_constraints(X_init_itern[0]):
                        continue
                X_init.append(X_init_itern)
                itern += 1
            X_init = torch.stack(X_init, 0).squeeze()
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
                    random_sample_within_discrete_tr_ordinal(centre, self.casmopolitan.length_init_discrete, self.config))
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            del X_tr_torch, fX_tr_torch, X_init_torch

        else:
            # If guided restart is not enabled, simply sample a number of points equal to the number of evaluated
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            # Sample Initial Points with frequency criterion
            self.X_init, itern = [], 0
            while(itern<self.casmopolitan.n_init):
                X_init = latin_hypercube(1, self.dim)
                X_init = from_unit_cube(X_init, self.lb, self.ub)
                if self.wrap_discrete:
                    X_init = self.warp_discrete(X_init, )
                X_init = onehot2ordinal(X_init, self.cat_dims)
                if self.cdr_constraints:
                    if not check_cdr_constraints(X_init[0]):
                        continue
                self.X_init.append(X_init)
                itern += 1
            self.X_init = torch.stack(self.X_init, 0).squeeze()

    def suggest(self, n_suggestions=1):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.casmopolitan.batch_size = n_suggestions
            # self.bo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.true_dim))
        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            if X_next.ndim==1:
                X_next = X_next[None, :]
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.casmopolitan._X) > 0:  # Use random points if we can't fit a GP
                X = deepcopy(self.casmopolitan._X)
                #fX = deepcopy(self.casmopolitan._fX).ravel()
                if self.kwargs['search_strategy'] in ['local']:
                    fX = copula_standardize(deepcopy(self.casmopolitan._fX).ravel())  # Use Copula
                else:
                    fX = deepcopy(self.casmopolitan._fX).ravel() # No need to use Copula as no GP predictions in here.

                # try:
                if True:
                    X_next[-n_adapt:, :] = self.casmopolitan._create_and_select_candidates(X, fX,
                                                                                           length=self.casmopolitan.length_discrete,
                                                                                           n_training_steps=500,
                                                                                          hypers={})[-n_adapt:, :]
                # except (ValueError, NanError, NotPSDError):
                # except:
                else:
                    print(f"Acquisition Failure with Kernel {self.casmopolitan.kernel_type}")
                    # if self.casmopolitan.kernel_type == 'ssk':
                    #     print(f"Trying with kernel {self.casmopolitan.kernel_type}")
                    #     self.casmopolitan.kernel_type = 'transformed_overlap'
                    #     X_next[-n_adapt:, :] = self.casmopolitan._create_and_select_candidates(X, fX,
                    #                                                                            length=self.casmopolitan.length_discrete,
                    #                                                                            n_training_steps=500,
                    #                                                                            hypers={})[-n_adapt:, :]
                    #     self.casmopolitan.kernel_type = 'ssk'
                    if self.casmopolitan.kernel_type in ['rbfBERT', 'cosineBERT']:
                        print("Random Acquisition")
                        X_random_next, j = [], 0
                        while (j < n_adapt):
                            X_next_j = latin_hypercube(1, self.dim)
                            X_next_j = from_unit_cube(X_next_j, self.lb, self.ub)
                            if self.wrap_discrete:
                                X_next_j = self.warp_discrete(X_next_j, )
                            X_next_j = onehot2ordinal(X_next_j, self.cat_dims)
                            if self.cdr_constraints:
                                if not check_cdr_constraints(X_next_j[0]):
                                    continue
                            X_random_next.append(X_next_j[0])
                            j += 1
                        X_random_next = np.stack(X_random_next, 0)
                        X_next[-n_adapt:, :] = X_random_next
                    else:
                        print('Resorting to Random Search')
                        # Create the initial population. Last column stores the fitness
                        X_next = np.random.randint(low=0, high=20, size=(n_suggestions, 11))

                        # Check for constraint violation
                        constraints_violated = np.logical_not(
                            check_constraint_satisfaction_batch(X_next))

                        # Continue until all samples satisfy the constraints
                        while np.sum(constraints_violated) != 0:
                            # Generate new samples for the ones that violate the constraints
                            X_next[constraints_violated] = np.random.randint(
                                low=0, high=20,
                                size=(np.sum(constraints_violated), 11))

                            # Check for constraint violation
                            constraints_violated = np.logical_not(
                                check_constraint_satisfaction_batch(X_next))
        suggestions = X_next
        return suggestions

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)
        # XX = torch.cat([ordinal2onehot(x, self.n_categories) for x in X]).reshape(len(X), -1)
        XX = X
        yy = np.array(y.detach().cpu())[:, None]
        # if self.wrap_discrete:
        #     XX = self.warp_discrete(XX, )

        if self.kwargs['search_strategy'] in ['local', 'batch_local']:
            if len(self.casmopolitan._fX) >= self.casmopolitan.n_init > 0:
                self.casmopolitan._adjust_length(yy)

        self.casmopolitan.n_evals += len(y)
        self.casmopolitan._X = np.vstack((self.casmopolitan._X, deepcopy(XX)))
        self.casmopolitan._fX = np.vstack((self.casmopolitan._fX, deepcopy(yy.reshape(-1, 1))))
        self.casmopolitan.X = np.vstack((self.casmopolitan.X, deepcopy(XX)))
        self.casmopolitan.fX = np.vstack((self.casmopolitan.fX, deepcopy(yy.reshape(-1, 1))))

        if self.kwargs['search_strategy'] in ['local', 'batch_local']:
            # Check for a restart
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
