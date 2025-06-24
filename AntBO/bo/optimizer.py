import os
from copy import deepcopy
from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.stats as ss
import torch
from gpytorch.utils.errors import NotPSDError, NanError

from bo.localbo_cat import CASMOPOLITANCat
from bo.localbo_utils import from_unit_cube, latin_hypercube, onehot2ordinal, \
    random_sample_within_discrete_tr_ordinal, check_cdr_constraints, space_fill_table_sample
from bo.utils import update_table_of_candidates, update_table_of_candidates_array
from utilities.constraint_utils import check_constraint_satisfaction_batch

COUNT_AA: int = 5


def order_stats(x: np.ndarray) -> np.ndarray:
    _, idx, cnt = np.unique(x, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x)  # Replace inf by something large
    assert x.ndim == 1 and np.all(np.isfinite(x))
    o_stats = order_stats(x=x)
    quantile = np.true_divide(o_stats, len(x) + 1)
    x_ss = ss.norm.ppf(quantile)
    return x_ss


class Optimizer:

    def __init__(self,
                 config: np.ndarray,
                 min_cuda,
                 normalise: bool = False,
                 cdr_constraints: bool = False,
                 n_init: int = None,
                 wrap_discrete: bool = True,
                 guided_restart: bool = True,
                 table_of_candidates: Optional[np.ndarray] = None,
                 table_of_candidate_embeddings: Optional[np.ndarray] = None,
                 embedding_from_array_dict: dict[str, np.ndarray] = None,
                 **kwargs) -> None:
        """Build wrapper class to use an optimizer in benchmark.

        Args:
            config: array. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
                being 2, 3, 4, and 5 respectively.
            guided_restart: whether to fit an auxiliary GP over the best points encountered in all previous restarts,
                and sample the points with maximum variance for the next restart.
            global_bo: whether to use the global version of the discrete GP without local modelling
            table_of_candidates: if not None, the suggestions should be taken from this list of candidates given as a
                                2d array of aas indices.
            table_of_candidate_embeddings: if not None, the embeddings of the candidates should be used to build the
                surrogate model
            embedding_from_array_dict: dictionary mapping the antibody arrays (as string of indices) to their embeddings
        """

        # Maps the input order.
        self.config = config.astype(int)
        self.kwargs = kwargs
        self.table_of_candidates = table_of_candidates
        self.table_of_candidate_embeddings = table_of_candidate_embeddings
        self.embedding_from_array_dict = embedding_from_array_dict
        if self.kwargs['kernel_type'] == 'ssk':
            assert self.kwargs['alphabet_size'] is not None
        # Number of one hot dimensions
        self.n_onehot = int(np.sum(config))
        # One-hot bounds
        self.lb = np.zeros(self.n_onehot)
        self.ub = np.ones(self.n_onehot)
        self.dim = len(self.lb)
        # True dim is simply the number of parameters (do not care about one-hot encoding etc).
        self.true_dim = len(config)
        if self.table_of_candidate_embeddings is not None:
            self.search_dim = self.table_of_candidate_embeddings.shape[1]
            self.lb = self.table_of_candidate_embeddings.min(0)
            self.ub = self.table_of_candidate_embeddings.max(0)
        else:
            self.search_dim = self.true_dim

        self.batch_size = None
        self.history = []
        self.wrap_discrete = wrap_discrete
        self.cat_dims = self.get_dim_info(config)
        self.cdr_constraints = cdr_constraints
        self.casmopolitan = CASMOPOLITANCat(
            min_cuda=min_cuda,
            dim=self.true_dim,
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,
            cdr_constraints=self.cdr_constraints,
            normalise=normalise,
            batch_size=1,  # We need to update this later
            verbose=False,
            config=self.config,
            **kwargs
        )

        # Our modification: define an auxiliary GP
        self.guided_restart = guided_restart
        # keep track of the best X and fX in each restart
        self.best_x_each_restart, self.best_fx_each_restart = None, None
        self.auxiliary_gp = None
        self.x_init, itern = [], 0

    def restart(self) -> None:
        from bo.gp import train_gp
        if self.guided_restart and len(self.casmopolitan.tr_fx) and self.kwargs['search_strategy'] == 'local':
            best_idx = self.casmopolitan.tr_fx.argmin()
            # Obtain the best X and fX within each restart (bo.tr_fx and bo.tr_x get erased at each restart,
            # but bo.x and bo.fx always store the full history
            if self.best_fx_each_restart is None:
                self.best_fx_each_restart = deepcopy(self.casmopolitan.tr_fx[best_idx])
                self.best_x_each_restart = deepcopy(self.casmopolitan.tr_x[best_idx]).reshape(1, -1)
            else:
                self.best_fx_each_restart = np.vstack(
                    (self.best_fx_each_restart, deepcopy(self.casmopolitan.tr_fx[best_idx])))
                self.best_x_each_restart = np.vstack(
                    (self.best_x_each_restart, deepcopy(self.casmopolitan.tr_x[best_idx])))

            if self.table_of_candidate_embeddings is not None:
                best_x_each_restart = np.array(
                    [self.embedding_from_array_dict[str(x.astype(int))] for x in self.best_x_each_restart]
                )
                kern = "mat52"
            else:
                best_x_each_restart = self.best_x_each_restart
                kern = "transformed_overlap"
            x_tr_torch = torch.tensor(best_x_each_restart, dtype=torch.float32).reshape(-1, self.search_dim)
            fx_tr_torch = torch.tensor(self.best_fx_each_restart, dtype=torch.float32).view(-1)

            # Train the auxiliary
            self.auxiliary_gp = train_gp(
                train_x=x_tr_torch, train_y=fx_tr_torch, use_ard=False, num_steps=300, kern=kern
            )
            # Generate random points in a Thompson-style sampling
            if self.table_of_candidates is not None:
                if self.table_of_candidate_embeddings is None:
                    candidates = self.table_of_candidates
                    metric = "hamming"
                else:
                    candidates = self.table_of_candidate_embeddings
                    metric = "euclidean"
                inds = space_fill_table_sample(
                    n_pts=min(self.casmopolitan.n_cand, len(self.table_of_candidates)),
                    table_of_candidates=candidates,
                    metric=metric
                )
                x_init_for_gp = deepcopy(candidates[inds])
                x_init = deepcopy(self.table_of_candidates[inds])
            else:
                x_init, itern = [], 0
                while itern < self.casmopolitan.n_cand:
                    x_init_itern = latin_hypercube(1, self.dim)
                    x_init_itern = from_unit_cube(x_init_itern, self.lb, self.ub)
                    if self.wrap_discrete:
                        x_init_itern = self.warp_discrete(x_init_itern)
                    x_init_itern = onehot2ordinal(x_init_itern, self.cat_dims)
                    if self.cdr_constraints:
                        if not check_cdr_constraints(x_init_itern[0]):
                            continue
                    x_init.append(x_init_itern)
                    itern += 1
                x_init = np.array(x_init)
                x_init_for_gp = x_init

            with torch.no_grad():
                self.auxiliary_gp.eval()
                x_init_for_gp_torch = torch.tensor(x_init_for_gp, dtype=torch.float32)
                # LCB-sampling
                posterior = self.auxiliary_gp(x_init_for_gp_torch)
                y_cand_mean = posterior.mean.cpu().detach().numpy()
                y_cand_var = posterior.variance.cpu().detach().numpy()
                y_cand = y_cand_mean - 1.96 * np.sqrt(y_cand_var)

            self.x_init = np.ones((self.casmopolitan.n_init, self.true_dim))
            indbest = np.argmin(y_cand)
            # The initial trust region centre for the new restart
            centre = deepcopy(x_init[indbest, :])
            # The centre is the first point to be evaluated
            self.x_init[0, :] = deepcopy(centre)

            if self.table_of_candidates is not None:  # sample from table
                n_sample = self.casmopolitan.n_init - 1
                # compute hamming distance with centre to restrict sampling to points around the center
                if self.table_of_candidate_embeddings is None:
                    metric = "hamming"
                    dist_to_centre = scipy.spatial.distance.cdist(self.table_of_candidates, centre.reshape(1, -1), metric)
                    dist_to_centre = dist_to_centre.flatten() * centre.shape[-1]
                    # discard center and points that are too far
                    if os.getenv("ANTBO_DEBUG", False):
                        upper_dist = len(centre)
                    else:
                        upper_dist = self.casmopolitan.length_init_discrete
                    candidates = self.table_of_candidates
                else:
                    metric = "euclidean"
                    dist_to_centre = scipy.spatial.distance.cdist(
                        self.table_of_candidate_embeddings,
                        self.embedding_from_array_dict[str(centre.flatten().astype(int))].reshape(1, -1),
                        metric
                    ).flatten()
                    upper_dist = self.casmopolitan.length_init * np.sqrt(self.search_dim)
                    candidates = self.table_of_candidate_embeddings

                filtr = np.logical_and(dist_to_centre <= upper_dist, upper_dist > 0)
                while filtr.sum() < n_sample:
                    upper_dist *= 1.2
                    filtr = np.logical_and(dist_to_centre <= upper_dist, upper_dist > 0)
                    if upper_dist > 1e30:
                        raise RuntimeError(f"There is a problem with the number of candidates? {len(candidates)}")
                candidates = candidates[filtr]

                # sample points
                inds = space_fill_table_sample(n_pts=n_sample, table_of_candidates=candidates, metric=metric)
                self.x_init[1:] = deepcopy(self.table_of_candidates[filtr][inds])
            else:
                for i in range(1, self.casmopolitan.n_init):
                    # Randomly sample within the initial trust region length around the centre
                    candidate = random_sample_within_discrete_tr_ordinal(
                        x_center=centre,
                        max_hamming_dist=self.casmopolitan.length_init_discrete,
                        n_categories=self.config
                    )
                    self.x_init[i] = deepcopy(candidate)
            self.casmopolitan.restart()
            self.casmopolitan.tr_x = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan.tr_fx = np.zeros((0, 1))
            del x_tr_torch, fx_tr_torch, x_init_for_gp_torch
        else:
            # If guided restart is not enabled, simply sample a number of points equal to the number of evaluated
            self.casmopolitan.restart()
            self.casmopolitan.tr_x = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan.tr_fx = np.zeros((0, 1))
            # If a table of candidates is available: use it
            if self.table_of_candidates is not None:
                if self.table_of_candidate_embeddings is not None:
                    metric = "euclidean"
                    candidates = self.table_of_candidate_embeddings
                else:
                    metric = "hamming"
                    candidates = self.table_of_candidates
                inds = space_fill_table_sample(
                    n_pts=self.casmopolitan.n_init, table_of_candidates=candidates, metric=metric
                )
                self.x_init = deepcopy(self.table_of_candidates[inds])
            else:  # Sample Initial Points with frequency criterion
                self.x_init, itern = [], 0
                while itern < self.casmopolitan.n_init:
                    x_init = latin_hypercube(1, self.dim)
                    x_init = from_unit_cube(x_init, self.lb, self.ub)
                    if self.wrap_discrete:
                        x_init = self.warp_discrete(x_init)
                    x_init = onehot2ordinal(x_init, self.cat_dims)
                    if self.cdr_constraints:
                        if not check_cdr_constraints(x_init[0]):
                            continue
                    self.x_init.append(x_init)
                    itern += 1
                self.x_init = np.array(self.x_init)

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """
        Args:
            n_suggestions: number of points to suggest
        """
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.casmopolitan.batch_size = n_suggestions
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])
            self.restart()

        x_next = np.zeros((n_suggestions, self.true_dim))

        table_of_candidates = deepcopy(self.table_of_candidates)
        table_of_candidate_embeddings = deepcopy(self.table_of_candidate_embeddings)

        # Pick from the initial points
        n_init = min(len(self.x_init), n_suggestions)
        if n_init > 0:
            x_next[:n_init] = deepcopy(self.x_init[:n_init, :])
            if x_next.ndim == 1:
                x_next = x_next[None, :]
            self.x_init = self.x_init[n_init:, :]  # Remove these pending points
            if table_of_candidates is not None:
                table_of_candidates, table_of_candidate_embeddings = update_table_of_candidates(
                    original_table=table_of_candidates,
                    observed_candidates=x_next[:n_init],
                    check_candidates_in_table=True,
                    table_of_candidate_embeddings=table_of_candidate_embeddings
                )

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if os.getenv("ANTBO_DEBUG", False):
            n_training_steps = 10
        else:
            n_training_steps = 500
        if n_adapt > 0:
            if len(self.casmopolitan.tr_x) > 0:  # Use random points if we can't fit a GP
                x = deepcopy(self.casmopolitan.tr_x)
                # fX = deepcopy(self.casmopolitan.tr_fx).ravel()
                if self.kwargs['search_strategy'] == 'local':
                    fx = copula_standardize(x=deepcopy(self.casmopolitan.tr_fx).ravel())  # Use Copula
                else:
                    fx = deepcopy(self.casmopolitan.tr_fx).ravel()  # No need to use Copula as no GP predictions in here.

                try:
                    new_candidates = self.casmopolitan.create_and_select_candidates(
                        x=x, fx=fx, n_training_steps=n_training_steps,
                        hypers={}, table_of_candidates=table_of_candidates,
                        table_of_candidate_embeddings=table_of_candidate_embeddings,
                        embedding_from_array_dict=self.embedding_from_array_dict
                    )
                    x_next[-n_adapt:, :] = new_candidates[-n_adapt:, :]
                except (ValueError, NanError, NotPSDError) as _:
                    print(f"Acquisition Failure with Kernel {self.casmopolitan.kernel_type}")
                    if self.casmopolitan.kernel_type == 'ssk':
                        print(f"Trying with kernel {self.casmopolitan.kernel_type}")
                        self.casmopolitan.kernel_type = 'transformed_overlap'
                        new_candidates = self.casmopolitan.create_and_select_candidates(
                            x=x, fx=fx, n_training_steps=n_training_steps,
                            hypers={}, table_of_candidates=table_of_candidates,
                            table_of_candidate_embeddings=table_of_candidate_embeddings,
                            embedding_from_array_dict=self.embedding_from_array_dict
                        )
                        x_next[-n_adapt:, :] = new_candidates[-n_adapt:, :]
                        self.casmopolitan.kernel_type = 'ssk'
                    elif self.casmopolitan.kernel_type in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
                        assert self.table_of_candidates is None, "not supported when given a table of candidates"
                        print("Random Acquisition")
                        x_random_next, j = [], 0
                        while j < n_adapt:
                            x_next_j = latin_hypercube(1, self.dim)
                            x_next_j = from_unit_cube(x_next_j, self.lb, self.ub)
                            if self.wrap_discrete:
                                x_next_j = self.warp_discrete(x_next_j)
                            x_next_j = onehot2ordinal(x_next_j, self.cat_dims)
                            if self.cdr_constraints:
                                if not check_cdr_constraints(x_next_j[0]):
                                    continue
                            x_random_next.append(x_next_j[0])
                            j += 1
                        x_random_next = np.stack(x_random_next, 0)
                        x_next[-n_adapt:, :] = x_random_next
                    else:
                        assert self.table_of_candidates is None, "not supported when given a table of candidates"
                        print('Resorting to Random Search')
                        # Create the initial population. Last column stores the fitness
                        x_next = np.random.randint(low=0, high=20, size=(n_suggestions, 11))

                        # Check for constraint violation
                        constraints_violated = np.logical_not(
                            check_constraint_satisfaction_batch(x_next))

                        # Continue until all samples satisfy the constraints
                        while np.sum(constraints_violated) != 0:
                            # Generate new samples for the ones that violate the constraints
                            x_next[constraints_violated] = np.random.randint(
                                low=0, high=20,
                                size=(np.sum(constraints_violated), 11))

                            # Check for constraint violation
                            constraints_violated = np.logical_not(
                                check_constraint_satisfaction_batch(x_next))
        suggestions = x_next
        return suggestions

    def observe(self, x: np.ndarray, y: torch.Tensor):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        x : array-like, shape (n, d)
        y : tensor of shape (n,)
        """
        assert len(x) == len(y), (len(x), len(y))
        yy = np.array(y.detach().cpu())[:, None]

        # check if some points are in x_init:
        if isinstance(self.x_init, np.ndarray):
            self.x_init = update_table_of_candidates_array(
                original_table=self.x_init,
                observed_candidates=x,
                check_candidates_in_table=False
            )
        else:
            assert len(self.x_init) == 0 or self.x_init is None

        if self.kwargs['search_strategy'] in ['local', 'batch_local']:
            if len(self.casmopolitan.tr_fx) >= self.casmopolitan.n_init > 0:
                self.casmopolitan.adjust_length(yy)

        self.casmopolitan.n_evals += len(x)
        self.casmopolitan.tr_x = np.vstack((self.casmopolitan.tr_x, deepcopy(x)))
        self.casmopolitan.tr_fx = np.vstack((self.casmopolitan.tr_fx, deepcopy(yy.reshape(-1, 1))))
        self.casmopolitan.x = np.vstack((self.casmopolitan.x, deepcopy(x)))
        self.casmopolitan.fx = np.vstack((self.casmopolitan.fx, deepcopy(yy.reshape(-1, 1))))
        if self.table_of_candidates is not None:
            print(f"Get {len(self.table_of_candidates)} candidate points"
                  f" in search table before observing new points")
            self.table_of_candidates, self.table_of_candidate_embeddings = update_table_of_candidates(
                original_table=self.table_of_candidates,
                observed_candidates=x,
                check_candidates_in_table=True,
                table_of_candidate_embeddings=self.table_of_candidate_embeddings
            )
            print(f"Get {len(self.table_of_candidates)} candidate points"
                  f" in search table after observing new points")

        if self.kwargs['search_strategy'] in ['local', 'batch_local']:
            # Check for a restart of the trust region
            min_cont_length_reached = self.casmopolitan.length <= self.casmopolitan.length_min
            if self.kwargs["kernel_type"] in ["transformed_overlap", "overlap"]:
                # only discrete variables
                min_cont_length_reached = False
            min_discr_length_reached = self.casmopolitan.length_discrete <= self.casmopolitan.length_min_discrete
            if min_cont_length_reached or min_discr_length_reached:
                self.restart()

    def warp_discrete(self, x: np.ndarray) -> np.ndarray:
        x_ = np.copy(x)
        # Process the integer dimensions
        if self.cat_dims is not None:
            for categorical_groups in self.cat_dims:
                max_col = np.argmax(x[:, categorical_groups], axis=1)
                x_[:, categorical_groups] = 0
                for idx, g in enumerate(max_col):
                    x_[idx, categorical_groups[g]] = 1
        return x_

    @staticmethod
    def get_dim_info(n_categories: np.ndarray[int]) -> list[list[int]]:
        dim_info = []
        offset = 0
        for i, cat in enumerate(n_categories):
            dim_info.append(list(range(offset, offset + cat)))
            offset += cat
        return dim_info
