# 2021.11.10-Add support for ssk
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import random
from collections import Callable
from copy import deepcopy

import logging
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import tqdm

from resources.casmopolitan.bo.kernels import *
import numpy as np


# debug
from resources.casmopolitan.bo.seq_kernel_fast import FastStringKernel


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


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, kern, likelihood,
                 outputscale_constraint,
                 ard_dims, cat_dims=None):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  # , cat_dims, int_dims)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, kern='transformed_overlap', hypers={},
             cat_dims=None, cont_dims=None,
             int_constrained_dims=None,
             noise_variance=None,
             cat_configs=None,
             **params):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    （train_x, train_y）: pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    int_constrained_dims: **Of the continuous dimensions**, which ones additionally are constrained to have integer
        values only?
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]
    # outputscale_constraint = Interval(0.05, 20.0)
    outputscale_constraint = Interval(0.5, 5.)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    # train_x = onehot2ordinal(train_x, cat_dims)
    ard_dims = None
    if use_ard:
        if 's-bert' in kern:
            assert 'embedding_bounds' in params, params
            ard_dims = params['embedding_bounds'].shape[1]
        else:
            ard_dims = train_x.shape[1]

    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap':
        kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'ssk': # subsequence string kernel
        kernel = FastStringKernel(seq_length=train_x.shape[1], alphabet_size=params['alphabet_size'],
                                  device=train_x.device)
    elif kern == 's-bert-matern52':
        assert 'input_transformation' in params, list(params.keys())
        assert 'embedding_bounds' in params, list(params.keys())
        kernel = WarpedMaternKernel(input_warping=params['input_transformation'],
                                    embedding_bounds=params['embedding_bounds'],
                                    nu=2.5, lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    elif kern == 's-bert-rbf':
        assert 'input_transformation' in params, list(params.keys())
        assert 'embedding_bounds' in params, list(params.keys())
        kernel = WarpedRBFKernel(input_warping=params['input_transformation'], embedding_bounds=params['embedding_bounds'],
                                 lengthscale_constraint=lengthscale_constraint,
                                 ard_num_dims=ard_dims)
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'mixed':
        assert cat_dims is not None and cont_dims is not None, 'cat_dims and cont_dims need to be specified if you wish' \
                                                               'to use the mix kernel'
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               integer_dims=int_constrained_dims,
                               **params)
    elif kern == 'mixed_overlap':
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               categorical_kern_type='overlap',
                               integer_dims=int_constrained_dims,
                               **params)
    else:
        raise ValueError('Unknown kernel choice %s' % kern)
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kernel,
        # lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        if not isinstance(kernel, FastStringKernel):
            hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

    trange = tqdm(range(num_steps), desc="GP fit")
    for _ in trange:
        optimizer.zero_grad()
        output = model(train_x, )
        loss = -mll(output, train_y).float()
        loss.backward()
        optimizer.step()
        trange.set_postfix({'loss': format(loss.item(), 'g')})

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


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


def local_search(x_center, f: Callable,
                 config,
                 max_hamming_dist,
                 n_restart: int = 1,
                 batch_size: int = 1,
                 step: int = 200):
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
            # print(acq_x, acq_neighbour)

            if acq_neighbour > acq_x:
                logging.info(''.join([str(int(i)) for i in neighbour.flatten()]) + ' ' + str(acq_neighbour))
                x = deepcopy(neighbour)
                # trajectory = np.vstack((trajectory, deepcopy(x)))
        logging.info('local search thread ended with highest acquisition %s' % acq_x)
        # print(compute_hamming_dist_ordinal(x_center, x, n_categories), acq_x)
        # print(x_center)
        return x, acq_x

    X = []
    fX = []
    for _ in tqdm(range(n_restart), desc='Local Search'):
        res = _ls(max_hamming_dist)
        X.append(res[0])
        fX.append(res[1])

    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    # select the top-k smallest
    # top_idices = np.argpartition(np.array(fX).flatten(), batch_size)[:batch_size]
    # print(np.array(fX).flatten()[top_idices])
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
        acq_x = f(x).detach().cpu().numpy()
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

            x_cont = x_cont_torch.detach().cpu().numpy()
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
                # acq_x = f(np.hstack((x_cat, x_cont))).detach().cpu().numpy()
                acq_neighbour = f(np.hstack((neighbour, x_cont))).detach().cpu().numpy()
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
