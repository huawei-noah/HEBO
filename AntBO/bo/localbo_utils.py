import logging
from itertools import groupby
from collections import Callable
import random
from copy import deepcopy

from bo.kernels import *
import re

COUNT_AA = 5
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: i for i, aa in enumerate(AA)}
idx_to_AA = {value: key for key, value in AA_to_idx.items()}
N_glycosylation_pattern = 'N[^P][ST][^P]'


# from Bio.SeqUtils.ProtParam import ProteinAnalysis


def check_cdr_constraints_all(x, x_center_local=None, hamming=None, config=None):
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

    if x_center_local is not None:
        # 1 if met (True)
        c4 = compute_hamming_dist_ordinal(x_center_local, x, config) <= hamming
        # Return 0 if True
        return int(not (c1)), int(not (c2)), int(not (c3)), int(not (c4))

    return int(not (c1)), int(not (c2)), int(not (c3))


def check_cdr_constraints(x) -> bool:
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


def sample_neighbour_ordinal_constrained(x, n_categories):
    """Same as above, but the variables are represented ordinally."""

    x_pert = deepcopy(x)
    n_categories = deepcopy(n_categories)
    # Chooose a variable to modify
    choice = random.randint(0, len(n_categories) - 1)
    # Obtain the current value.
    curr_val = x[choice]
    options = [i for i in range(n_categories[choice]) if i != curr_val]
    random.shuffle(options)
    i = 0
    x_pert[choice] = options[i]

    while np.logical_not(check_cdr_constraints(x_pert)) and i < (len(n_categories) - 1):
        i += 1
        x_pert[choice] = options[i]
    return x_pert


def neighbourhood_init(x_center_local, config, pop_size):
    pop = np.array([sample_neighbour_ordinal_constrained(x_center_local, config) for _ in range((pop_size))])
    pop[0] = x_center_local
    return pop


def random_sample_within_discrete_tr(x_center, max_hamming_dist, categorical_dims,
                                     mode='ordinal'):
    """Randomly sample a point within the discrete trust region"""
    if max_hamming_dist < 1:  # Normalised hamming distance is used
        bit_change = int(max_hamming_dist * len(categorical_dims))
    else:  # Hamming distance is not normalized
        max_hamming_dist = min(max_hamming_dist, len(categorical_dims))
        bit_change = int(max_hamming_dist)

    x_pert = deepcopy(x_center)
    # Randomly sample n bits to change.
    modified_bits = random.sample(range(len(categorical_dims)), bit_change)
    for bit in modified_bits:
        n_values = len(categorical_dims[bit])
        # Change this value
        selected_value = random.choice(range(n_values))
        # Change to one-hot encoding
        substitute_values = np.array([1 if i == selected_value else 0 for i in range(n_values)])
        x_pert[categorical_dims[bit]] = substitute_values
    return x_pert


def random_sample_within_discrete_tr_ordinal(x_center, max_hamming_dist, n_categories):
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
from pymoo.factory import get_mutation, get_crossover, get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
from torch.quasirandom import SobolEngine


class CDRH3Prob(Problem):
    """CDRH3 Problem For pymoo.
    Maximise f_acq but taking the negative in  _evaluate to perform minimisation overall.
    A solution is considered as feasible of all constraint violations are less than zero."""

    def __init__(self, f_acq: Callable, n_var=11, n_obj=1, n_constr=3, xl=0, xu=19, cdr_constraints=True,
                 device=torch.device('cpu'), dtype=torch.float32, f2_acq=None, f3_acq=None):
        if f2_acq is not None:
            n_obj += 1
        if f3_acq is not None:
            n_obj += 1
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr if cdr_constraints else 0,
                         xl=xl,
                         xu=xu)

        self.f2_acq = f2_acq
        self.f3_acq = f3_acq
        self.f_acq = f_acq
        self.cdr_constraints = cdr_constraints
        self.device = device
        self.dtype = dtype

    def _evaluate(self, x, out, *args, **kwargs):
        with torch.no_grad():
            # Switch from max to min problem
            acq_x = -1.0 * self.f_acq(x).detach().cpu().numpy()
        out["F"] = acq_x

        if self.cdr_constraints:
            c1, c2, c3 = zip(*list(map(lambda seq: check_cdr_constraints_all(seq), x)))
            out["G"] = np.column_stack([c1, c2, c3])


class CDRH3ProbHamming(CDRH3Prob):
    """CDRH3 Problem For pymoo.
    Maximise f_acq but taking the negative in  _evaluate to perform minimisation overall.
    A solution is considered as feasible of all constraint violations are less than zero."""

    def __init__(self, max_hamming_distance=0, x_center_local=None, config=None, **kwargs):
        super().__init__(**kwargs)
        self.hamming = max_hamming_distance
        self.x_center_local = x_center_local
        self.config = config

    def _evaluate(self, x, out, *args, **kwargs):
        # Always 1 Objective
        with torch.no_grad():
            # Switch from max to min problem
            acq_x = -1.0 * self.f_acq(x).detach().cpu().numpy()
            if self.f2_acq is not None:
                acq2_x = -1.0 * self.f2_acq(x).detach().cpu().numpy()
                acq_x = np.column_stack([acq_x, acq2_x])
            if self.f3_acq is not None:
                acq3_x = -1.0 * self.f3_acq(x).detach().cpu().numpy()
                acq_x = np.column_stack([acq_x, acq3_x])
        out["F"] = acq_x

        if self.cdr_constraints:
            c1, c2, c3, c4 = zip(*list(
                map(lambda seq: check_cdr_constraints_all(seq, x_center_local=self.x_center_local, hamming=self.hamming,
                                                          config=self.config), x)))
            out["G"] = np.column_stack([c1, c2, c3, c4])


def get_pop(seq_len, pop_size, x_center_local, seed=0):
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


def glocal_search(x_center,
                  f: Callable,
                  config,
                  max_hamming_dist,
                  cdr_constraints: bool = False,
                  n_restart: int = 1,
                  n_obj=1,
                  batch_size: int = 1,
                  seed=0,
                  seq_len=11,
                  dtype=torch.float32,
                  device=torch.device('cpu'),
                  pop_size: int = 200,
                  eliminate_duplicates=True,
                  biased=True,
                  f2: Callable = None,
                  f3: Callable = None,
                  **kwargs):
    """
    Global & Glocal search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :return:
    """
    if kwargs['kernel_type'] == 'ssk':
        pop_size = 20

    x_center_local = deepcopy(x_center)
    if biased:
        # True, Do neighbourhood sampling
        init_pop = neighbourhood_init(x_center_local, config, pop_size)
    else:
        # False, Do Global sampling
        init_pop = get_pop(seq_len, pop_size, x_center_local, seed=seed)

    if f2 is not None or f3 is not None or n_obj > 1:
        eliminate_duplicates = False

    algorithm = NSGA2(pop_size=pop_size,
                      n_offsprings=pop_size,
                      sampling=init_pop,
                      # crossover=get_crossover("int_sbx", eta=15, prob=0.0, prob_per_variable=0.0),
                      crossover=get_crossover("int_sbx", eta=15, prob=0.9, prob_per_variable=1.0 / seq_len),
                      mutation=get_mutation("int_pm", eta=20),
                      eliminate_duplicates=eliminate_duplicates)
    problem = CDRH3Prob(n_obj=n_obj, n_var=seq_len, f_acq=f, f2_acq=f2, f3_acq=f3, cdr_constraints=cdr_constraints,
                        device=device, dtype=dtype)
    termination = get_termination("n_gen", seq_len * kwargs['alphabet_size'])

    res = minimize(problem,
                   algorithm,
                   termination,
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
    # Turn back to maximise problem
    fX = -1.0 * F
    # Remove constraint violated
    fX = fX[~G.any(1)]

    if not eliminate_duplicates:
        # Remove duplocates
        X, idX = np.unique(X, axis=0, return_index=True)
        # Remove duplocates
        fX = fX[idX]

    if X.ndim == 1:
        X = X[None, :]
        fX = fX[None, :]

    if f2 is not None or f3 is not None or n_obj > 1:
        # Sample from pareto front
        idx = np.random.randint(0, X.shape[0], batch_size)
        X_next = X[idx]
        acq_next = np.array(fX).flatten()[idx]
    else:
        # Selects top batchsize from list
        top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
        X_next = np.array([x for i, x in enumerate(X) if i in top_idices])
        acq_next = np.array(fX).flatten()[top_idices]

    return X_next, acq_next


def blocal_search(x_center,
                  f: Callable,
                  config,
                  max_hamming_dist,
                  cdr_constraints: bool = False,
                  n_restart: int = 1,
                  batch_size: int = 1,
                  seed=0,
                  seq_len=11,
                  n_obj=1,
                  dtype=torch.float32,
                  device=torch.device('cpu'),
                  pop_size: int = 200,
                  eliminate_duplicates=True,
                  f2: Callable = None,
                  f3: Callable = None,
                  **kwargs):
    """
    Batch Local search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :return:
    """
    if kwargs['kernel_type'] == 'ssk':
        pop_size = 20

    if f2 is not None or f3 is not None or n_obj > 1:
        eliminate_duplicates = False

    x_center_local = deepcopy(x_center)
    init_pop = neighbourhood_init(x_center_local, config, pop_size)
    algorithm = NSGA2(pop_size=pop_size,
                      n_offsprings=pop_size,
                      sampling=init_pop,
                      crossover=get_crossover("int_sbx", eta=15, prob=0.9, prob_per_variable=1.0 / seq_len),
                      # crossover=get_crossover("int_sbx", eta = 15, prob = 0.0, prob_per_variable=0.0),
                      mutation=get_mutation("int_pm", eta=20),
                      eliminate_duplicates=eliminate_duplicates)
    problem = CDRH3ProbHamming(n_obj=n_obj, x_center_local=x_center_local, n_constr=4, config=config,
                               max_hamming_distance=max_hamming_dist, n_var=seq_len, f_acq=f, f2_acq=f2, f3_acq=f3,
                               cdr_constraints=cdr_constraints, device=device, dtype=dtype)
    termination = get_termination("n_gen", seq_len * kwargs['alphabet_size'])

    res = minimize(problem,
                   algorithm,
                   termination,
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
    # Turn back to maximise problem
    fX = -1.0 * F
    # Remove constraint violated
    fX = fX[~G.any(1)]

    if X.ndim == 1:
        X = X[None, :]
        fX = fX[None, :]

    if not eliminate_duplicates:
        # Remove duplocates
        X, idX = np.unique(X, axis=0, return_index=True)
        # Remove duplocates
        fX = fX[idX]

    if f2 is not None or f3 is not None or n_obj > 1:
        idx = np.random.randint(0, X.shape[0], batch_size)
        X_next = X[idx]
        acq_next = np.array(fX).flatten()[idx]
    else:
        # Selects top batchsize from list
        top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
        X_next = np.array([x for i, x in enumerate(X) if i in top_idices])
        acq_next = np.array(fX).flatten()[top_idices]
    return X_next, acq_next


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
                 **kwargs):
    """
    Local search algorithm
    :param n_restart: number of restarts
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)
    :param max_hamming_dist: maximum Hamming distance from x_center
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :return:
    """

    def _ls(hamming):
        """One restart of local search"""
        # x0 = deepcopy(x_center)
        x_center_local = deepcopy(x_center)
        tol = 100
        trajectory = np.array([x_center_local])
        x = x_center_local

        acq_x = f(x).detach().cpu().numpy()
        for i in range(step):
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
