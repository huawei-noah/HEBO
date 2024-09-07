import logging
import random
import re
from collections import Callable
from copy import deepcopy
from itertools import groupby

import scipy
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from tqdm import tqdm

from bo.kernels import *

# from Bio.SeqUtils.ProtParam import ProteinAnalysis

COUNT_AA = 5
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: i for i, aa in enumerate(AA)}
idx_to_AA = {value: key for key, value in AA_to_idx.items()}
N_glycosylation_pattern = 'N[^P][ST][^P]'


def check_cdr_constraints_all(x):
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
    if (charge > 2.0 or charge < -2.0):
        c2 = False
    else:
        c2 = True
    if re.search(N_glycosylation_pattern, x_to_seq):
        c3 = False
    else:
        c3 = True

    # stability = prot.instability_index()
    # if stability>40:
    #    return False
    # If constraint is satisfied return 0 for pymoo
    return int(not (c1)), int(not (c2)), int(not (c3))


def check_cdr_constraints(x):
    constr = check_cdr_constraints_all(x)
    return not np.any(constr)


def onehot2ordinal(x, categorical_dims):
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


def ordinal2onehot(x, n_categories):
    """Convert ordinal to one-hot"""
    res = np.zeros(np.sum(n_categories))
    offset = 0
    for i, cat in enumerate(n_categories):
        res[offset + int(x[i])] = 1
        offset += cat
    return torch.tensor(res)


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    # random.seed(random.randint(0, 1e6))
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]
    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert

    return X


def space_fill_table_sample(n_pts: int, table_of_candidates: np.ndarray) -> np.ndarray:
    """
    Sample points from a table of candidates

    Args:
        n_pts: number of points to sample
        table_of_candidates: 2d array from which to sample points

    Returns:
        samples: 2d array with shape (n_pts, n_dim) taken from table_of_candidates
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
        distances = scipy.spatial.distance.cdist(table_of_candidates, candidates[:next_ind], metric="hamming")
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
        candidates[next_ind:] = table_of_candidates[
            np.random.choice(remaining_inds, size=len(candidates[next_ind:]), replace=False)]
    return deepcopy(candidates)


def compute_hamming_dist(x1, x2, categorical_dims, normalize=False):
    """
    Compute the hamming distance of two one-hot encoded strings.
    :param x1:
    :param x2:
    :param categorical_dims: list of lists. e.g.
    [[1, 2], [3, 4, 5, 6]] where idx 1 and 2 correspond to the first variable, and
    3, 4, 5, 6 coresponds to the second variable with 4 possible options
    :return:
    """
    dist = 0.
    for i, var_groups in enumerate(categorical_dims):
        if not np.all(x1[var_groups] == x2[var_groups]):
            dist += 1.
    if normalize:
        dist /= len(categorical_dims)
    return dist


def compute_hamming_dist_ordinal(x1, x2, n_categories=None, normalize=False):
    """Same as above, but on ordinal representations."""
    hamming = (x1 != x2).sum()
    if normalize:
        return hamming / len(x1)
    return hamming


def sample_neighbour(x, categorical_dims):
    """Sample a neighbour (i.e. of unnormalised Hamming distance of 1) from x"""
    x_pert = deepcopy(x)
    # Sample a variable where x_pert will differ from the selected sample
    # random.seed(random.randint(0, 1e6))
    choice = random.randint(0, len(categorical_dims) - 1)
    # Change the value of that variable randomly
    var_group = categorical_dims[choice]
    # Confirm what is value of the selected variable in x (we will not sample this point again)
    for var in var_group:
        if x_pert[var] != 0:
            break
    value_choice = random.choice(var_group)
    while value_choice == var:
        value_choice = random.choice(var_group)
    x_pert[var] = 0
    x_pert[value_choice] = 1
    return x_pert


def sample_neighbour_ordinal(x, n_categories):
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


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
from torch.quasirandom import SobolEngine


class CDRH3Prob(Problem):
    """CDRH3 Problem For pymoo.
    Maximise f_acq but taking the negative in  _evaluate to perform minimisation overall.
    A solution is considered as feasible of all constraint violations are less than zero."""

    def __init__(self, f_acq: Callable, n_var=11, n_obj=1, n_constr=3, xl=0, xu=19, cdr_constraints=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr if cdr_constraints else 0,
                         xl=xl,
                         xu=xu)
        self.f_acq = f_acq
        self.cdr_constraints = cdr_constraints
        self.device = device
        self.dtype = dtype

    def _evaluate(self, x, out, *args, **kwargs):
        with torch.no_grad():
            X = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
            # Switch from max to min problem
            acq_x = -1.0 * self.f_acq(X).detach().cpu().numpy()
        out["F"] = acq_x

        if self.cdr_constraints:
            c1, c2, c3 = zip(*list(map(lambda seq: check_cdr_constraints_all(seq), x)))
            out["G"] = np.column_stack([c1, c2, c3])


def get_biased_pop(seq_len, pop_size, x_center_local, seed=0):
    eng = SobolEngine(seq_len, scramble=True, seed=seed)
    sobol_samp = eng.draw(pop_size)
    # sobol_samp = sobol_samp * (space.opt_ub - space.opt_lb) + space.opt_lb
    sobol_samp = sobol_samp * 19
    sobol_samp = sobol_samp.int().numpy()
    # set initial point as current best guess, so mutations etc can do local or global search
    sobol_samp[0] = x_center_local
    return sobol_samp


def test_f(x):
    return torch.from_numpy(np.random.randint(0, 20, x.shape[0]))


def local_table_search(x_center: np.ndarray,
                       f: Callable,
                       config: np.ndarray,
                       table_of_candidates: np.ndarray,
                       batch_size: int,
                       max_hamming_dist,
                       dtype=torch.float32,
                       device=torch.device('cpu'),
                       max_batch_size=5000, **kwargs):
    """
    Search strategy:
        1. Compute filtr of valid points around center
        2. Iteratively jump to a new point that is still in the valid area

    Args:
        x_center: 1d array corresponding to center of search space in transformed space
        config: the config for the categorical variables (number of categoories per dim)

    Returns:
        x_candidates: 2d ndarray acquisition points
        f_x_candidates: 1d array function value of these points
    """
    assert x_center.ndim == 1, x_center.shape
    assert table_of_candidates.ndim == 2, table_of_candidates.shape
    assert batch_size == 1, "Methods is designed to output only one candidate for now"
    n_candidates = 0
    hamming_dists = scipy.spatial.distance.cdist(table_of_candidates, x_center.reshape(1, -1), metric="hamming")
    hamming_dists *= x_center.shape[-1]  # denormalize
    hamming_dists = hamming_dists.flatten()

    # entry `i` contains the number of points in the table of candidates that are at distance `i` of the center
    n_cand_per_dist = np.array([(hamming_dists == i).sum() for i in range(table_of_candidates.shape[-1])])
    max_hamming_dist = max(max_hamming_dist, np.argmax(np.cumsum(n_cand_per_dist) > 0))

    table_of_candidates = table_of_candidates[hamming_dists <= max_hamming_dist]

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
            n_ones = max_batch_size - cand_filtr.sum()
            n_zeros = new_filtr.sum() - n_ones
            if cand_filtr.sum() + new_filtr.sum() > max_batch_size:
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

    return current_center.reshape(1, -1), np.array([fmax])


def glocal_search(x_center,
                  f: Callable,
                  config,
                  max_hamming_dist,
                  cdr_constraints: bool = False,
                  n_restart: int = 1,
                  batch_size: int = 1,
                  seed=0,
                  seq_len=11,
                  dtype=torch.float32,
                  device=torch.device('cpu'),
                  pop_size: int = 200,
                  table_of_candidates=None):
    """
    Local search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :param table_of_candidates: search within a table of candidates (not supported for now)
    :return:
    """
    assert table_of_candidates is None
    x_center_local = deepcopy(x_center)
    init_pop = get_biased_pop(seq_len, pop_size, x_center_local, seed=seed)
    crossover = SBX(prob=0.9, prob_var=1.0 / seq_len, eta=15, repair=RoundingRepair(), vtype=float)
    mutation = PolynomialMutation(prob=1.0, eta=20, repair=RoundingRepair())
    algorithm = NSGA2(pop_size=pop_size,
                      n_offsprings=200,
                      sampling=init_pop,
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=False)
    problem = CDRH3Prob(n_var=seq_len, f_acq=f, cdr_constraints=cdr_constraints, device=device, dtype=dtype)
    termination = ("n_gen", 11 * 20)

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   seed=seed,
                   verbose=False)

    # Make sure to filter any that are not satisfied
    if res.G.ndim == 1:
        G = res.G[None, :]
        X = res.X[None, :]
        F = res.F[None, :]
    else:
        G = res.G
        X = res.X
        F = res.F

    # Remove constraint violated
    X = X[~G.any(1)]
    # Remove duplocates
    X = np.unique(X, axis=0)

    # Turn back to maximise problem
    fX = -1.0 * F

    # Selects top batchsize from list
    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]


def local_search(x_center,
                 f: Callable,
                 config,
                 max_hamming_dist,
                 cdr_constraints: bool = False,
                 n_restart: int = 1,
                 batch_size: int = 1,
                 seed=0,
                 dtype=torch.float32,
                 device=torch.device('cpu'),
                 step: int = 200,
                 table_of_candidates=None):
    """
    Local search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :param table_of_candidates: search within a table of candidates (not supported for now)
    :return:
    """
    assert table_of_candidates is None

    def _ls(hamming):
        """One restart of local search"""
        # x0 = deepcopy(x_center)
        x_center_local = deepcopy(x_center)
        tol = 100
        trajectory = np.array([x_center_local])
        x = x_center_local

        acq_x = f(x).detach().cpu().numpy()
        for _ in tqdm(range(step), f"local search to acquire point {i + 1}"):
            tol_ = tol
            is_valid = False
            while not is_valid:
                neighbour = sample_neighbour_ordinal(x, config)
                if cdr_constraints:
                    if not check_cdr_constraints(neighbour):
                        continue
                if 0 < compute_hamming_dist_ordinal(x_center_local, neighbour, config) <= hamming \
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
                logging.info(''.join([str(int(i)) for i in neighbour.flatten()]) + ' ' + str(acq_neighbour))
                x = deepcopy(neighbour)
        logging.info('local search thread ended with highest acquisition %s' % acq_x)
        return x, acq_x

    X = []
    fX = []
    for i in range(n_restart):
        res = _ls(max_hamming_dist)
        X.append(res[0])
        fX.append(res[1])

    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]


def interleaved_search(x_center, f: Callable,
                       cat_dims,
                       cont_dims,
                       config,
                       ub,
                       lb,
                       max_hamming_dist,
                       n_restart: int = 1,
                       batch_size: int = 1,
                       interval: int = 1,
                       step: int = 200):
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

    def _interleaved_search(x0):
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        x_cat, x_cont = x[cat_dims], x[cont_dims]
        n_step = 0
        while n_step <= step:
            # First optimise the continuous part, freezing the categorical part
            def f_cont(x_cont_):
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
                while not is_valid:
                    neighbour = sample_neighbour_ordinal(x_cat, config)
                    if 0 <= compute_hamming_dist_ordinal(x_center[cat_dims], neighbour, config) <= max_hamming_dist:
                        is_valid = True
                    else:
                        tol_ -= 1
                if tol_ < 0:
                    logging.info("Tolerance exhausted on this local search thread.")
                    break
                # acq_x = f(np.hstack((x_cat, x_cont))).detach().numpy()
                acq_neighbour = f(np.hstack((neighbour, x_cont))).detach().numpy()
                if acq_neighbour > acq_x:
                    x_cat = deepcopy(neighbour)
                    acq_x = acq_neighbour
            # print(x_cat, x_cont, acq_x)
            n_step += interval

        x = np.hstack((x_cat, x_cont))
        return x, acq_x

    X, fX = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]


if __name__ == ' __main__ ':
    x_center_local = np.random.randint(0, 19, 11)
    X_next, acq_next = glocal_search(x_center_local, test_f, {'seed': 0}, 2, 3, 2, True, 11)
