import logging
import random
import re
from copy import deepcopy
from itertools import groupby
from typing import Callable, Optional
from typing import Literal

import numpy as np
import scipy
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from torch.quasirandom import SobolEngine
from tqdm import tqdm

from bo.kernels import *

ACQ_FUNCTIONS = Literal['ucb', 'ei', 'thompson', 'eiucb', 'mace', 'imace']
SEARCH_STRATS = Literal['glocal', 'local', 'local-no-hamming', 'batch_local', 'global']

COUNT_AA = 5
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: i for i, aa in enumerate(AA)}
idx_to_AA = {value: key for key, value in AA_to_idx.items()}
N_glycosylation_pattern = 'N[^P][ST][^P]'


def check_cdr_constraints_all(x: np.ndarray) -> tuple[int, int, int]:
    # Constraints on CDR3 sequence
    x_to_seq = ''.join(idx_to_AA[int(aa)] for aa in x)
    # prot = ProteinAnalysis(x_to_seq)
    # charge = prot.charge_at_pH(7.4)
    # Counting number of consecutive keys
    count = max([sum(1 for _ in group) for _, group in groupby(x_to_seq)])
    if count > 5:
        c1 = False
    else:
        c1 = True
    charge = 0
    for char in x_to_seq:
        charge += int(char == 'R' or char == 'K') + 0.1 * int(char == 'H') - int(char == 'D' or char == 'E')
    if charge > 2.0 or charge < -2.0:
        c2 = False
    else:
        c2 = True
    if re.search(N_glycosylation_pattern, x_to_seq):
        c3 = False
    else:
        c3 = True

    # If constraint is satisfied return 0 for pymoo
    return int(not c1), int(not c2), int(not c3)


def check_cdr_constraints(x: np.ndarray) -> bool:
    constr = check_cdr_constraints_all(x=x)
    return not np.any(constr)


def onehot2ordinal(x: np.ndarray, categorical_dims: list[list[int]]) -> torch.tensor:
    """Convert one-hot representation of strings back to ordinal representation."""
    from itertools import chain
    if x.ndim == 1:
        x = x.reshape(1, -1)
    categorical_dims_flattned = list(chain(*categorical_dims))
    # Select those categorical dimensions only
    x = x[:, categorical_dims_flattned]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    res = torch.zeros(x.shape[0], len(categorical_dims), dtype=torch.float32)
    for i, var_group in enumerate(categorical_dims):
        res[:, i] = torch.argmax(x[:, var_group], dim=-1).float()
    return res


def from_unit_cube(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts: int, dim: int) -> np.ndarray:
    """Basic Latin hypercube implementation with center perturbation."""
    x = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    # random.seed(random.randint(0, 1e6))
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        x[:, i] = centers[np.random.permutation(n_pts)]
    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    x += pert

    return x


def space_fill_table_sample(n_pts: int, table_of_candidates: np.ndarray, metric: str) -> np.ndarray:
    """
    Sample points from a table of candidates

    Args:
        n_pts: number of points to sample
        table_of_candidates: 2d array from which to sample points
        metric: metric to use to get distance among points

    Returns:
        indices: array of shape (n_pts,) corresponding to selected elements in table_of_candidates
    """
    if n_pts == 0:
        raise ValueError("n_pts should be strictly greater than 0")
    if len(table_of_candidates) == 0:
        raise ValueError("table_of_candidates should contain candidates")
    if len(table_of_candidates) < n_pts:
        raise ValueError(
            f"table_of_candidates should contain at least {n_pts} candidates, got {len(table_of_candidates)}"
        )
    selected_inds = set()
    candidates = np.zeros((n_pts, table_of_candidates.shape[-1]))
    ind = np.random.randint(0, len(table_of_candidates))
    selected_inds.add(ind)
    next_ind = 0
    candidates[next_ind] = table_of_candidates[ind]  # sample first point at random
    next_ind += 1
    for _ in range(1, min(n_pts, 100)):  # sample the first 100 points with a space-filling strategy
        # compute distance among table_of_candidates and already_selected candidates
        distances = scipy.spatial.distance.cdist(table_of_candidates, candidates[:next_ind], metric=metric)
        distances[distances == 0] = -np.inf  # penalize already selected points
        mean_dist = distances.mean(-1)
        max_mean_dist = mean_dist.max()
        # sample among best
        ind = np.random.choice(np.arange(len(table_of_candidates))[mean_dist == max_mean_dist])
        selected_inds.add(ind)
        candidates[next_ind] = table_of_candidates[ind]
        next_ind += 1
    if next_ind < n_pts:  # add points randomly among the ones that has not been picked yet
        remaining_inds = [ind for ind in range(len(table_of_candidates)) if ind not in selected_inds]
        selected_inds.update(np.random.choice(remaining_inds, size=len(candidates[next_ind:]), replace=False))
    return np.array(list(selected_inds))


def compute_hamming_dist(x1: np.ndarray, x2: np.ndarray, categorical_dims: list[list[[int]]], normalize: bool = False) \
        -> float:
    """
    Compute the hamming distance of two one-hot encoded strings.
    Args:
        x1: ...
        x2: ...
        categorical_dims: list of lists. e.g. [[1, 2], [3, 4, 5, 6]] where idx 1 and 2 correspond to the first variable,
         and 3, 4, 5, 6 coresponds to the second variable with 4 possible options
        normalize: whehther to divide the distance by the number of dimensions

    Returns:
        hamming_dist: the hamming distance between x1 and x2
    """
    dist = 0.
    for i, var_groups in enumerate(categorical_dims):
        if not np.all(x1[var_groups] == x2[var_groups]):
            dist += 1.
    if normalize:
        dist /= len(categorical_dims)
    return dist


def compute_hamming_dist_ordinal(x1: np.ndarray, x2: np.ndarray, normalize: bool = False) -> float:
    """Same as above, but on ordinal representations."""
    hamming = (x1 != x2).sum()
    if normalize:
        return hamming / len(x1)
    return hamming


def sample_neighbour_ordinal(x: np.ndarray, n_categories: list[int]) -> np.ndarray:
    """Same as above, but the variables are represented ordinally."""
    x_pert = deepcopy(x)
    # Chooose a variable to modify
    choice = random.randint(0, len(n_categories) - 1)
    # Obtain the current value.
    curr_val = x[choice]
    options = [i for i in range(n_categories[choice]) if i != curr_val]
    x_pert[choice] = random.choice(options)
    return x_pert


def random_sample_within_discrete_tr_ordinal(x_center: np.ndarray, max_hamming_dist, n_categories) -> np.ndarray:
    """Same as above, but here we assume a ordinal representation of the categorical variables."""
    # random.seed(random.randint(0, 1e6))
    if max_hamming_dist < 1:
        bit_change = int(max(max_hamming_dist * len(n_categories), 1))
    else:
        bit_change = int(min(max_hamming_dist, len(n_categories)))
    x_pert = deepcopy(x_center)
    modified_bits = random.sample(range(len(n_categories)), bit_change)
    for bit in modified_bits:
        options = np.arange(n_categories[bit])
        x_pert[bit] = int(random.choice(options))
    return x_pert


class CDRH3Prob(Problem):
    """CDRH3 Problem For pymoo.
    Maximise f_acq but taking the negative in  _evaluate to perform minimisation overall.
    A solution is considered as feasible of all constraint violations are less than zero."""

    def __init__(self, f_acq: Callable, n_var: int = 11, n_obj: int = 1, n_constr: int = 3, xl: int = 0, xu: int = 19,
                 cdr_constraints: bool = True, device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr if cdr_constraints else 0,
                         xl=xl,
                         xu=xu)
        self.f_acq = f_acq
        self.cdr_constraints = cdr_constraints
        self.device = device
        self.dtype = dtype

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        with torch.no_grad():
            x_torch = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
            # Switch from max to min problem
            acq_x = -1.0 * self.f_acq(x_torch).detach().cpu().numpy()
        out["F"] = acq_x

        if self.cdr_constraints:
            c1, c2, c3 = zip(*list(map(lambda seq: check_cdr_constraints_all(seq), x)))
            out["G"] = np.column_stack([c1, c2, c3])


def get_biased_pop(seq_len: int, pop_size: int, x_center_local: np.ndarray, seed: int = 0) -> np.ndarray:
    eng = SobolEngine(seq_len, scramble=True, seed=seed)
    sobol_samp = eng.draw(pop_size)
    # sobol_samp = sobol_samp * (space.opt_ub - space.opt_lb) + space.opt_lb
    sobol_samp = sobol_samp * 19
    sobol_samp = sobol_samp.int().numpy()
    # set initial point as current best guess, so mutations etc can do local or global search
    sobol_samp[0] = x_center_local
    return sobol_samp


def test_f(x: np.ndarray) -> torch.tensor:
    return torch.from_numpy(np.random.randint(0, 20, x.shape[0]))


def local_table_search(
        x_center: np.ndarray, f: Callable,
        table_of_candidates: np.ndarray,
        batch_size: int, max_hamming_dist: float,
        max_batch_size: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Search strategy:
        1. Compute filtr of valid points around center
        2. Iteratively jump to a new point that is still in the valid area

    Args:
        x_center: 1d array corresponding to center of search space in transformed space
        f: function to optimize
        table_of_candidates: table of candidates to search in
        batch_size: number of points to suggest as a result of the optimization
        max_hamming_dist: max distance between the center and the suggested points
        max_batch_size: maximum number of points that can be passed to `f` at once.

    Returns:
        x_candidates: 2d ndarray acquisition points
        f_x_candidates: 1d array function value of these points
    """
    print(f"-- Start local table search (Max hamming: {max_hamming_dist}, "
          f"number of candidates: {len(table_of_candidates)}) --")

    assert x_center.ndim == 1, x_center.shape
    assert table_of_candidates.ndim == 2, table_of_candidates.shape
    assert batch_size == 1, "Methods is designed to output only one candidate for now"
    hamming_dists = scipy.spatial.distance.cdist(table_of_candidates, x_center.reshape(1, -1), metric="hamming")
    hamming_dists *= x_center.shape[-1]  # denormalize
    hamming_dists = hamming_dists.flatten()

    # entry `i` contains the number of points in the table of candidates that are at distance `i` of the center
    n_cand_per_dist = np.array([(hamming_dists == i).sum() for i in range(table_of_candidates.shape[-1] + 1)])
    max_hamming_dist = max(max_hamming_dist, np.argmax(np.cumsum(n_cand_per_dist) > 0))

    table_of_candidates = table_of_candidates[hamming_dists <= max_hamming_dist]
    print(f"---> Will search candidates with at most distance "
          f"{max_hamming_dist} among {len(table_of_candidates)} candidates")

    fmax = -np.inf

    current_center = deepcopy(x_center)

    for _ in tqdm(range(10)):
        if len(table_of_candidates) == 0:
            break

        # gather points from closest to farthest
        hamming_dists = scipy.spatial.distance.cdist(table_of_candidates, current_center.reshape(1, -1),
                                                     metric="hamming")
        hamming_dists *= current_center.shape[-1]
        hamming_dists = hamming_dists.flatten()

        cand_filtr = np.zeros(len(table_of_candidates)).astype(bool)
        dist_to_current_center = 1
        while cand_filtr.sum() < max_batch_size and dist_to_current_center <= table_of_candidates.shape[-1]:
            new_filtr = hamming_dists == dist_to_current_center
            if cand_filtr.sum() + new_filtr.sum() > max_batch_size:
                n_ones = max_batch_size - cand_filtr.sum()
                n_zeros = new_filtr.sum() - n_ones
                new_filtr[new_filtr] = np.random.permutation(
                    np.concatenate([np.zeros(n_zeros), np.ones(n_ones)]).astype(bool))
            cand_filtr = cand_filtr + new_filtr
            dist_to_current_center += 1

        if cand_filtr.sum() == 0:  # no more candidates to evaluate
            break

        # evaluate acquisition function
        acq_x = f(table_of_candidates[cand_filtr]).detach().cpu().numpy().flatten()
        if acq_x.max() > fmax:  # new best
            fmax = acq_x.max()
            current_center = table_of_candidates[cand_filtr][np.argmax(acq_x)]

        # remove already evaluated points from table of candidates
        table_of_candidates = table_of_candidates[~cand_filtr]
    print("-- End of the local search --")
    return current_center.reshape(1, -1), np.array([fmax])


def local_table_emmbedding_search(
        x_center: np.ndarray, f: Callable,
        table_of_candidates: np.ndarray,
        table_of_candidate_embeddings: Optional[np.ndarray],
        embedding_from_array_dict: dict[str, np.ndarray],
        batch_size: int, per_dim_max_dist: float,
        max_batch_size=5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Search strategy:
        1. Compute filtr of valid points around center
        2. Iteratively jump to a new point that is still in the valid area

    Args:
        x_center: 1d array corresponding to center of search space in index space
        f: function to optimize
        table_of_candidates: table of candidates to search in
        batch_size: number of points to suggest as a result of the optimization
        per_dim_max_dist: per-dimension distance between the center embedding and the suggested points
        max_batch_size: maximum number of points that can be passed to `f` at once.
        table_of_candidate_embeddings: if not None, the embeddings of the candidates should be used to build the
                surrogate model
        embedding_from_array_dict: dictionary mapping the antibody arrays (as string of indices) to their embeddings

    Returns:
        x_candidates: 2d ndarray acquisition points
        f_x_candidates: 1d array function value of these points
    """
    print(f"-- Start local table search (number of candidates: {len(table_of_candidates)}) --")

    assert x_center.ndim == 1, x_center.shape
    assert table_of_candidates.ndim == 2, table_of_candidates.shape
    if len(table_of_candidates) != len(table_of_candidate_embeddings):
        raise ValueError(f"{len(table_of_candidates)}, {len(table_of_candidate_embeddings)}")
    assert batch_size == 1, "Methods is designed to output only one candidate for now"

    x_center_embedding = embedding_from_array_dict[str(x_center.astype(int))]
    per_dim_dist = np.abs(x_center_embedding - table_of_candidate_embeddings)

    filtr = (per_dim_dist <= per_dim_max_dist).all(axis=1)
    while filtr.sum() < batch_size:
        per_dim_max_dist *= 1.1
        filtr = (per_dim_dist <= per_dim_max_dist).all(axis=1)

    print(f"---> Will search candidates among {filtr.sum()} candidates near the center")
    inds_to_evaluate = np.arange(len(table_of_candidate_embeddings))[filtr]

    best_ind = 0
    fmax = -np.inf

    min_i = 0
    while min_i < len(inds_to_evaluate):
        batch_inds = inds_to_evaluate[min_i:min_i + max_batch_size]
        try:
            acq_x = f(table_of_candidate_embeddings[batch_inds]).detach().cpu().numpy().flatten()
            min_i += max_batch_size
        except RuntimeError as e:
            if 'CUDA error: out of memory' in str(e):
                print("CUDA out of memory error caught!")
                torch.cuda.empty_cache()
                max_batch_size //= 2
                if max_batch_size < 1:
                    raise
                continue
            else:
                raise

        if acq_x.max() > fmax:  # new best
            fmax = acq_x.max()
            best_ind = batch_inds[np.argmax(acq_x)]

    print("-- End of the local search --")
    return table_of_candidates[best_ind].reshape(1, -1), np.array([fmax])


def glocal_search(x_center: np.ndarray,
                  f: Callable,
                  cdr_constraints: bool = False,
                  batch_size: int = 1,
                  seed=0,
                  seq_len=11,
                  dtype=torch.float32,
                  device=torch.device('cpu'),
                  pop_size: int = 200,
                  table_of_candidates=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Local search algorithm

    Args:
        x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
        f: the function handle to evaluate x on (the acquisition function, in this case)
        cdr_constraints: whether constraints should be checked
        batch_size: number of points to suggest as a result of the optimization
        seed: random seed
        seq_len: length of sequence
        dtype: torch dtype
        device: torch device
        pop_size: population size for the genetic algorithm
        table_of_candidates: search within a table of candidates (not supported for now)

    Returns:
        suggestions: result of the optimization
        f_suggestions: function values of the suggested points
    """
    if table_of_candidates is not None:
        raise NotImplementedError()
    x_center_local = deepcopy(x_center)
    init_pop = get_biased_pop(seq_len, pop_size, x_center_local, seed=seed)
    crossover = SBX(prob=0.9, prob_var=1.0 / seq_len, eta=15, repair=RoundingRepair(), vtype=float)
    mutation = PolynomialMutation(prob=1.0, eta=20, repair=RoundingRepair())
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=200,
        sampling=init_pop,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=False
    )
    problem = CDRH3Prob(n_var=seq_len, f_acq=f, cdr_constraints=cdr_constraints, device=device, dtype=dtype)
    termination = ("n_gen", 11 * 20)

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   seed=seed,
                   verbose=False)

    # Make sure to filter any that are not satisfied
    if res.G.ndim == 1:
        g = res.G[None, :]
        x = res.X[None, :]
        f_x = res.F[None, :]
    else:
        g = res.G
        x = res.X
        f_x = res.F

    # Remove constraint violated
    x = x[~g.any(1)]
    # Remove duplocates
    x = np.unique(x, axis=0)

    # Turn back to maximise problem
    f_x = -1.0 * f_x

    # Selects top batchsize from list
    top_indices = np.argpartition(np.array(f_x).flatten(), -batch_size)[-batch_size:]
    return np.array([xx for i, xx in enumerate(x) if i in top_indices]), np.array(-f_x).flatten()[top_indices]


def local_search(x_center: np.ndarray,
                 f: Callable,
                 config: list[int],
                 max_hamming_dist: int,
                 cdr_constraints: bool = False,
                 n_restart: int = 1,
                 batch_size: int = 1,
                 step: int = 200,
                 table_of_candidates=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Local search algorithm

    Args:
        x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
        f: the function handle to evaluate x on (the acquisition function, in this case)
        max_hamming_dist: maximum Hamming distance from x_center
        n_restart: number of restarts
        config: list of number of categories per dim
        cdr_constraints: whether constraints should be checked
        batch_size: number of points to suggest as a result of the optimization
        step: number of maximum local search steps the algorithm is allowed to take.
        table_of_candidates: search within a table of candidates (not supported for now)

    Returns:
        suggestions: result of the optimization
        f_suggestions: function values of the suggested points
    """
    assert table_of_candidates is None

    def _ls(hamming: int, n_pnt: int) -> tuple[np.ndarray, np.ndarray]:
        """One restart of local search"""
        # x0 = deepcopy(x_center)
        x_center_local = deepcopy(x_center)
        tol = 100
        trajectory = np.array([x_center_local])
        x = x_center_local

        acq_x = f(x).detach().cpu().numpy()
        for _ in tqdm(range(step), f"local search to acquire point {n_pnt + 1}"):
            tol_ = tol
            is_valid = False
            neighbour = None
            while not is_valid:
                neighbour = sample_neighbour_ordinal(x=x, n_categories=config)
                if cdr_constraints:
                    if not check_cdr_constraints(neighbour):
                        continue
                if 0 < compute_hamming_dist_ordinal(x1=x_center_local, x2=neighbour) <= hamming \
                        and not any(np.equal(trajectory, neighbour).all(1)):
                    is_valid = True
                else:
                    tol_ -= 1
            if tol_ < 0:
                logging.info("Tolerance exhausted on this local search thread.")
                return x, acq_x

            acq_x = f(x).detach().cpu().numpy()
            acq_neighbour = f(neighbour).detach().cpu().numpy()

            if acq_neighbour > acq_x:
                logging.info(''.join([str(int(j)) for j in neighbour.flatten()]) + ' ' + str(acq_neighbour))
                x = deepcopy(neighbour)
        logging.info('local search thread ended with highest acquisition %s' % acq_x)
        return x, acq_x

    optima = []
    f_optima = []
    for i in range(n_restart):
        res = _ls(hamming=max_hamming_dist, n_pnt=i)
        optima.append(res[0])
        f_optima.append(res[1])

    top_indices = np.argpartition(np.array(f_optima).flatten(), -batch_size)[-batch_size:]
    return np.array([x for j, x in enumerate(optima) if j in top_indices]), np.array(f_optima).flatten()[top_indices]


def interleaved_search(
        x_center, f: Callable, cat_dims, cont_dims, config, ub, lb, max_hamming_dist, n_restart: int = 1,
        batch_size: int = 1, interval: int = 1, step: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interleaved search combining both first-order gradient-based method on the continuous variables and the local search
    for the categorical variables.
    Parameters
    ----------
    x_center: the starting point of the search
    cat_dims: the indices of the categorical dimensions
    cont_dims: the indices of the continuous dimensions
    f: function handle (normally this should be the acquisition function)
    config: the config for the categorical variables
    lb: lower bounds (trust region boundary) for the continuous variables
    ub: upper bounds (trust region boundary) for the continuous variables
    max_hamming_dist: maximum hamming distance boundary (for the categorical variables)
    n_restart: number of restarts of the optimisaiton
    batch_size:
    interval: number of steps to switch over (to start with, we optimise with n_interval steps on the continuous
        variables via a first-order optimiser, then we switch to categorical variables (with the continuous ones fixed)
        and etc.
    step: maximum number of search allowed.

    Returns
    -------

    """
    # todo: the batch setting needs to be changed. For the continuous dimensions, we cannot simply do top-n indices.

    from torch.quasirandom import SobolEngine

    # select the initialising points for both the continuous and categorical variables and then hstack them together
    # x0_cat = np.array([deepcopy(sample_neighbour_ordinal(x_center[cat_dims], config)) for _ in range(n_restart)])
    x0_cat = np.array([deepcopy(random_sample_within_discrete_tr_ordinal(x_center[cat_dims], max_hamming_dist, config))
                       for _ in range(n_restart)])
    # x0_cat = np.array([deepcopy(x_center[cat_dims]) for _ in range(n_restart)])
    seed = np.random.randint(int(1e6))
    sobol = SobolEngine(len(cont_dims), scramble=True, seed=seed)
    x0_cont = sobol.draw(n_restart).cpu().detach().numpy()
    x0_cont = lb + (ub - lb) * x0_cont
    x0 = np.hstack((x0_cat, x0_cont))
    tol = 100
    lb, ub = torch.tensor(lb, dtype=torch.float32), torch.tensor(ub, dtype=torch.float32)

    def _interleaved_search(x0_: torch.tensor) -> tuple[np.ndarray, np.ndarray]:
        x = deepcopy(x0_)
        acq_x = f(x).detach().numpy()
        x_cat, x_cont = x[cat_dims], x[cont_dims]
        n_step = 0
        while n_step <= step:
            # First optimise the continuous part, freezing the categorical part
            def f_cont(x_cont_: torch.tensor) -> torch.tensor:
                """The function handle for continuous optimisation"""
                x_ = torch.cat((x_cat_torch, x_cont_)).float()
                return -f(x_)

            x_cont_torch = torch.tensor(x_cont, dtype=torch.float32).requires_grad_(True)
            x_cat_torch = torch.tensor(x_cat, dtype=torch.float32)
            optimizer = torch.optim.Adam([{"params": x_cont_torch}], lr=0.1)
            for _ in range(interval):
                optimizer.zero_grad()
                acq = f_cont(x_cont_torch).float()
                try:
                    acq.backward()
                    # print(x_cont_torch, acq, x_cont_torch.grad)
                    optimizer.step()
                except RuntimeError:
                    print('Exception occured during backpropagation. NaN encountered?')
                    pass
                with torch.no_grad():
                    # Ugly way to do clipping
                    x_cont_torch.data = torch.max(torch.min(x_cont_torch, ub), lb)

            x_cont = x_cont_torch.detach().numpy()
            del x_cont_torch

            # Then freeze the continuous part and optimise the categorical part
            for j in range(interval):
                is_valid = False
                tol_ = tol
                neighbour = None
                while not is_valid:
                    neighbour = sample_neighbour_ordinal(x_cat, config)
                    if 0 <= compute_hamming_dist_ordinal(x1=x_center[cat_dims], x2=neighbour) <= max_hamming_dist:
                        is_valid = True
                    else:
                        tol_ -= 1
                if tol_ < 0:
                    logging.info("Tolerance exhausted on this local search thread.")
                    break
                acq_neighbour = f(np.hstack((neighbour, x_cont))).detach().numpy()
                if acq_neighbour > acq_x:
                    x_cat = deepcopy(neighbour)
                    acq_x = acq_neighbour
            n_step += interval

        x = np.hstack((x_cat, x_cont))
        return x, acq_x

    xx, f_xx = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0_=x0[i, :])
        xx.append(res[0])
        f_xx.append(res[1])
    top_indices = np.argpartition(np.array(f_xx).flatten(), -batch_size)[-batch_size:]
    return np.array([x for i, x in enumerate(xx) if i in top_indices]), np.array(f_xx).flatten()[top_indices]
