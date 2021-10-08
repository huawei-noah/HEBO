import time
from typing import Type, Optional, List, Any, Dict, Union, Tuple

import numpy as np
import torch
from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement, \
    qProbabilityOfImprovement, AcquisitionFunction, OneShotAcquisitionFunction, qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import gen_batch_initial_conditions, ExpMAStoppingCriterion, optimize_acqf
from botorch.optim.parameter_constraints import _arrayify, make_scipy_bounds
from botorch.optim.utils import columnwise_clamp, _filter_kwargs, fix_features
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError
from scipy.optimize import minimize
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler

from core.comp_acquisition.compositional_acquisition import CompositionalAcquisition
from core.es.evolution_opt import EvolutionOpt, DEopt, CMAESopt
from core.gp.custom_gp import SingleTaskRoundGP
from core.params_helper import ParamSpace
from core.utils.utils_query import query_AcqFunc, query_scheduler, query_optimizer, \
    query_covar
from custom_optimizer import ASCGD
from custom_optimizer.comp_opt import CompositionalOptimizer
from custom_optimizer.utils.utils import columnwise_clamp_


class BayesOptimization:
    """ Class to handle botorch bayesian optimisation of a black-box function

        Args:
            params_h: parameter space
            negate: if True, consider minimizing balck-box instead of maximizing
            optimizer: name of the optimizer to use (Adam, SGD, CAdam...)
            optimizer_kwargs: kwargs for optimizer
            scheduler (Optional): scheduler for the optimizer
            acq_func: name of the botorch acquisition function to use (qExpectedImprovement,...)
            acq_func_kwargs (Optional): kwargs for acquisition function
            initial_design_numdata: Number of points randomly picked to initialize hyperparameters tuning via BO
            num_MC_samples_acq: number of samples for MC acquisition loss estimation
            num_raw_samples: number of raw starts considered among which `num_starts` will be selected
            num_starts: number of starts for optimization of acquisition function
            num_opt_steps: number of optimization steps
            scheduler_kwargs: string-specified dictionary for scheduler
            verbose: verbosity level
            seed: seed for the experiment
            covar: name of the botorch kernel used for the GP
            covar_kw: kernel kwargs
            time_limit_per_acq_step: limit on execution time for
    """

    def __init__(self,
                 params_h: ParamSpace,
                 negate: bool,
                 optimizer: str,
                 acq_func: str,
                 scheduler: Optional[str] = None,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 acq_func_kwargs: Optional[Dict[str, Any]] = None,
                 initial_design_numdata: int = 3,
                 num_MC_samples_acq: int = 256,
                 num_raw_samples: int = 512,
                 num_starts: int = 64,
                 num_opt_steps: int = 128,
                 scheduler_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: float = 0,
                 seed: int = 0,
                 noise_free: bool = False,
                 covar: str = 'matern-5/2',
                 covar_kw: Dict[str, Any] = None,
                 time_limit_per_acq_step: float = np.inf,
                 early_stop: bool = False,
                 int_mask: Optional[List[int]] = None,
                 device: Optional[int] = None
                 ):

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
        }
        self.int_mask = int_mask
        self.negate = negate
        self.noise_free = noise_free
        self.covar = covar.lower()
        if covar_kw is None:
            covar_kw = {}
        self.covar_kw = covar_kw
        self.scale_covar: bool = covar_kw.pop('scale', True)

        self.params_space = params_h
        self.tensor_bounds: Tensor = torch.tensor(self.params_space.get_array_bounds()).to(**self.tkwargs)

        self.num_acq_steps = 0
        self.num_MC_samples_acq = num_MC_samples_acq
        # define acquisition function

        self.AcqFunc = query_AcqFunc(acq_func.split('-')[0])
        self.acq_func_kwargs = {} if acq_func_kwargs is None else acq_func_kwargs.copy()
        self.initial_design_numdata = initial_design_numdata
        self.num_restarts = num_starts

        self.optimizer_kwargs: Dict[str, Any] = {}
        self.num_opt_steps = num_opt_steps
        self.early_stop = early_stop
        self.opt_name = optimizer
        if self.opt_name == 'RandomSearch':
            self.opt = optimizer
            self.opt_name = optimizer
        elif self.opt_name == 'LBFGSB':
            self.opt = 'L-BFGS-B'
            self.opt_name = optimizer
        elif self.opt_name == 'LBFGSB-nested':
            self.opt = 'L-BFGS-B'
            self.opt_name = optimizer
            self.acq_func_kwargs['fixed_z'] = True
            self.acq_func_kwargs['K_g'] = self.num_MC_samples_acq
            self.acq_func_kwargs['m'] = self.num_MC_samples_acq
        else:
            self.opt = query_optimizer(optimizer.split('-')[0])
            if optimizer_kwargs is not None:
                self.optimizer_kwargs = optimizer_kwargs.copy()
            if issubclass(self.opt, CompositionalOptimizer):
                # `-ME` stands for Memory Efficient, if so `z` samples are not sampled before-hand and for computation
                # of `g` new samples are drawn
                mem_efficient = '-ME' in optimizer
                if mem_efficient:
                    print('Memory efficient setup')
                self.acq_func_kwargs['fixed_z'] = not mem_efficient
                self.acq_func_kwargs['K_g'] = self.num_MC_samples_acq
                self.acq_func_kwargs['m'] = self.num_MC_samples_acq
                self.acq_func_kwargs['approx'] = mem_efficient
                if not mem_efficient:
                    self.acq_func_kwargs['m'] *= self.num_opt_steps

            if 'qFiniteSum' in acq_func:
                self.acq_func_kwargs['K_g'] = self.num_MC_samples_acq
                self.num_MC_samples_acq *= self.num_opt_steps

            if issubclass(self.opt, EvolutionOpt):
                assert 'pop' not in self.optimizer_kwargs or self.num_restarts == self.optimizer_kwargs['pop'], \
                    'Population size argument `pop` in `optimizer_kwargs` must be the same as num_restarts'

        self.resampler = SobolQMCNormalSampler(num_samples=self.num_MC_samples_acq,
                                               resample=self.opt not in ['RandomSearch', 'L-BFGS-B'])
        self.scheduler_class: Optional[Type[_LRScheduler]] = query_scheduler(scheduler)
        self.scheduler_kwargs: Dict[str, Any] = {}
        if scheduler_kwargs is not None:
            self.scheduler_kwargs = scheduler_kwargs

        if not isinstance(self.opt, str):
            if 'nested' not in self.opt_name:
                if issubclass(self.opt, CompositionalOptimizer) != issubclass(self.AcqFunc, CompositionalAcquisition):
                    raise ValueError(
                        f"Optimizer and Acquisition function should have same compatibility with compositional"
                        f" optimization but we have optimizer that is"
                        f" {'' if issubclass(self.opt, CompositionalOptimizer) else 'not'}"
                        f" compositional and Acquisition function that is "
                        f"{'' if issubclass(self.AcqFunc, CompositionalAcquisition) else 'not'} compositional")
            else:
                if not issubclass(self.AcqFunc, CompositionalAcquisition) or issubclass(self.opt,
                                                                                        CompositionalOptimizer):
                    raise ValueError(
                        f"To run Nested MC you must choose a non-compositional optimizer "
                        f"and a compositional acquisition function")

        self.num_raw_samples = num_raw_samples
        self.time_limit_per_acq_step = time_limit_per_acq_step
        self.time_per_acq_step: List[float] = []

        self.verbose = verbose

        # run bayesian optimization routine
        self.data_X: Optional[Tensor] = None
        self.data_Y: Optional[Tensor] = None

        self.execution_times_s: List[float] = []
        self.acq_step_time_ref: float = time.time()
        self.total_ex_time: float = 0

        # seed for reproducibility
        self.seed = seed

    @property
    def input_dim(self) -> int:
        return self.params_space.d

    @property
    def num_points(self):
        return 0 if self.data_Y is None else len(self.data_Y)

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.resampler.seed = seed

    def reset(self):
        self.num_acq_steps = 0

        self.time_per_acq_step: List[float] = []

        # run bayesian optimization routine
        self.data_X: Optional[Tensor] = None
        self.data_Y: Optional[Tensor] = None

        self.execution_times_s = []
        self.total_ex_time = 0

    @property
    def fact(self) -> int:
        return -1 if self.negate else 1

    def gen(self, n_suggestions: int = 1, real: bool = True) -> Union[List[Dict[str, Any]], np.ndarray]:
        """ Suggest `n_suggestions` new acquisition points

        Args:
            n_suggestions: number of new points to acquire
            real: whether to return suggestions in the real param space or in the search space

        Returns:
            suggestions: either a list of points lying in real space (each expressed as dictionary) or a
                         `n_suggestion x num_params`-array
        """
        t = time.time()
        num_acquired = 0
        candidates: Optional[np.ndarray] = None
        if self.num_points < self.initial_design_numdata:
            if self.data_X is None:
                candidates = np.atleast_2d(self.params_space.get_random_search_point(n_suggestions))
            else:
                candidates = np.atleast_2d(self.params_space.get_random_search_point(min(
                    self.initial_design_numdata - self.num_points, n_suggestions
                )))
            num_acquired = candidates.shape[0]

        if num_acquired < n_suggestions:
            aux_candidates: np.ndarray = self.one_acq_step(q=n_suggestions - num_acquired)
            if candidates is not None:
                candidates = np.vstack([candidates, aux_candidates])
            else:
                candidates = np.atleast_2d(aux_candidates)
        if real:
            candidates: List[Dict[str, Any]] = [self.params_space.get_real_params(candidate) for candidate in
                                                candidates]
        self.num_acq_steps += 1
        self.execution_times_s.append(time.time() - t)
        self.total_ex_time += time.time() - t
        return candidates

    def observe(self, X_search: Union[np.ndarray, List[Dict[str, Any]]], y: Union[List[float], np.ndarray]) -> None:
        """ Observe new points and add them to current dataset

        Args:
            X_search: newly evaluated points
            y: value obtained when evaluating the black-box at `X_search` points
        """
        new_X: Tensor = torch.tensor(X_search).to(**self.tkwargs) if isinstance(X_search, np.ndarray) else torch.tensor(
            list(
                map(lambda params: list(params.values()), X_search)), **self.tkwargs)
        if not isinstance(y, Tensor):
            y = torch.tensor(y)
        new_Y: Tensor = self.fact * y.to(**self.tkwargs)
        filter_nan = torch.isnan(new_Y)
        filter_inf = torch.isinf(new_Y)
        filter_all = filter_inf + filter_nan
        if filter_all.sum() > 0:
            new_X = new_X[~filter_all]
            new_Y = new_Y[~filter_all]
        new_Y.unsqueeze_(1)
        if self.data_X is None:
            self.data_X = new_X
            self.data_Y = new_Y
        else:
            self.data_X = torch.cat([self.data_X, new_X])
            self.data_Y = torch.cat([self.data_Y, new_Y])
        if self.verbose > 0:
            print(f"Best after observation of {len(self.data_Y)} points: {self.fact * self.data_Y.max().item():g}")

    def get_normalisation_el(self):
        """ Get scale and offset factors for normalised inputs """
        scale = 1 / (self.tensor_bounds[1] - self.tensor_bounds[0])
        offset = - scale * self.tensor_bounds[0]
        return scale, offset

    def one_acq_step(self, q: int) -> np.ndarray:
        """
        Perform one acquisition step

        Args:
            q: number of points to acquire

        Returns:
            new_X: a `q x d` numpy array of acquired points
        """
        self.acq_step_time_ref: float = time.time()

        outer_dim = 1  # real-valued objective function

        best_candidates, best_values = [self.data_X[torch.argmax(self.data_Y)].clone()], [torch.max(self.data_Y).item()]

        # prepare data
        train_X_it = normalize(self.data_X, self.tensor_bounds)
        train_Y_it = self.data_Y

        acq_bounds = torch.stack(
            [torch.zeros(self.params_space.d, **self.tkwargs), torch.ones(self.params_space.d, **self.tkwargs)])
        # fit surrogate model given the data
        if self.int_mask is None or len(self.int_mask) == 0:
            # no need to consider a mixed GP handling integer values
            covar_module = query_covar(self.covar, train_X_it, train_Y_it, self.scale_covar, **self.covar_kw)
            model = SingleTaskGP(train_X_it, train_Y_it, outcome_transform=Standardize(outer_dim),
                                 covar_module=covar_module)
        else:
            print(f'Use round GP ({len(self.int_mask)} integer variables)')
            scale, offset = self.get_normalisation_el()
            model = SingleTaskRoundGP(train_X_it, train_Y_it, normalization_scale=scale, normalization_offset=offset,
                                      int_mask=self.int_mask, outcome_transform=Standardize(outer_dim))
        if self.noise_free:
            model.likelihood.noise = 2e-4
            model.likelihood.raw_noise.requires_grad = False
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        self.acq_func_kwargs["sampler"] = self.resampler
        if issubclass(self.AcqFunc, (qExpectedImprovement, qProbabilityOfImprovement)):
            self.acq_func_kwargs["best_f"] = best_values[-1]
        if issubclass(self.AcqFunc, qNoisyExpectedImprovement):
            self.acq_func_kwargs["X_baseline"] = train_X_it
        if issubclass(self.AcqFunc, qUpperConfidenceBound):
            delta = self.acq_func_kwargs.get('delta', None)
            if delta is not None:
                self.acq_func_kwargs['beta'] = max(0, 4 / np.pi * np.log(
                    self.num_points ** (self.input_dim / 2 + 2) * np.pi ** 2 / (3 * delta)))

        acq_func = self.AcqFunc(model, **_filter_kwargs(self.AcqFunc, **self.acq_func_kwargs))

        if self.opt == 'RandomSearch':
            # For fairness, number of acquisition function evaluations should be the same when using selected optimizer
            # and when using Random Search tm(though, due to scaling issues, they may not be evaluated all at once)
            t_random_batch = self.num_restarts * self.num_opt_steps + self.num_raw_samples
            q_q_fantasies = q
            if isinstance(acq_func, OneShotAcquisitionFunction):
                q_q_fantasies += acq_func.num_fantasies
            new_X_candidates = torch.rand(t_random_batch, q_q_fantasies, self.input_dim, **self.tkwargs)
            acq_func_estimates: Tensor = torch.zeros(t_random_batch)
            slice_size = 1024
            for t in range(t_random_batch // slice_size + 1):
                slice_t = slice(t * slice_size, min(t_random_batch, (t + 1) * slice_size))
                if slice_t.stop - slice_t.start == 0:
                    continue
                acq_func_estimates[slice_t] = acq_func(new_X_candidates[slice_t])
            new_X: Tensor = new_X_candidates[torch.argmax(acq_func_estimates), :q]  # shape q x dim
        elif self.opt == 'L-BFGS-B':
            new_X = self.bfgs_optimize_acqf_and_get_observation(acq_func, q=q, bounds=acq_bounds)
        elif issubclass(self.opt, EvolutionOpt):
            new_X = self.es_optimize_acqf_and_get_observation(acq_func, q=q, bounds=acq_bounds)
        else:
            new_X, meta_dic = self.optimize_acqf_and_get_observation(acq_func, q=q, bounds=acq_bounds)

        new_X = new_X.detach()  # shape q_q_fantasies x dim
        new_X = unnormalize(new_X, self.tensor_bounds)
        self.time_per_acq_step.append(time.time() - self.acq_step_time_ref)
        return new_X.detach().cpu().numpy()

    def optimize_acqf_and_get_observation(self, acq_func: AcquisitionFunction, bounds: Tensor,
                                          q: int) -> Tuple[Tensor, Dict[str, Any]]:
        """ Optimize acquisition function

        Args:
            acq_func:  The acquisition function to optimize
            q:  number of new acquisition points we look for
            bounds:  A `2 x d` tensor of lower and upper bounds for each column of `X`.

        Returns:
             X tensor of shape q x d best candidate maximize of acquisition function
             meta_dic : dictionary that may contains entries:
        """
        # we'll want gradients for the input
        q_q_fantasies = q  # q + q_fantaisies when using Knowledge Gradient
        if isinstance(acq_func, OneShotAcquisitionFunction):
            q_q_fantasies += acq_func.num_fantasies
        try:
            X: Tensor = gen_batch_initial_conditions(acq_func, bounds, q_q_fantasies, self.num_restarts,
                                                     self.num_raw_samples,
                                                     options={'seed': self.seed + self.num_points})
        except (RuntimeError, NotPSDError) as e:
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print(f'{e.args[0][:13]} error handled during intitialization')
            X: Tensor = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(self.num_restarts, q_q_fantasies,
                                                                         self.input_dim, **self.tkwargs)

        assert X.shape == (self.num_restarts, q_q_fantasies, bounds.shape[
            -1]), f"X.shape should be {(self.num_restarts, q_q_fantasies, bounds.shape[-1])} but got {X.shape}"
        X.requires_grad_(True)
        X_copy = X.detach().clone()

        # set parameters to optimize
        params = (dict(params=X),)
        if isinstance(acq_func, CompositionalAcquisition) and 'nested' not in self.opt_name:
            with torch.no_grad():
                try:
                    # initialize `Y` to E(g(X))
                    Y: Tensor = acq_func.inner_g_expected(X.clone())
                except (RuntimeError, NotPSDError) as e:
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    print(f'{e} error in initializing Y')
                    Y: Tensor = torch.zeros(size=(X.shape[0], X.shape[1], acq_func.get_m())).to(**self.tkwargs)
            Y.requires_grad_(True)
            params += (dict(params=Y),)
        optimizer = self.opt(params, **self.optimizer_kwargs)
        scheduler = None
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)

        losses_step = np.inf * torch.ones(self.num_restarts).to(**self.tkwargs)
        stopping_criterion = ExpMAStoppingCriterion(maxiter=self.num_opt_steps)
        stop: bool = False
        time_one_opt_step: float = 0
        new_samples = True
        i = 0
        # run a basic optimization loop
        for i in range(self.num_opt_steps):
            time_one_opt_step_ref: float = time.time()
            # check whether we have time to run a new optimization step
            if (time.time() - self.acq_step_time_ref) + 5 * time_one_opt_step > self.time_limit_per_acq_step:
                break
            optimizer.zero_grad()

            msg = ''
            if isinstance(acq_func, CompositionalAcquisition) and issubclass(self.opt, CompositionalOptimizer):
                eval_J = False  # whether to evaluate f(g(X)) during `opt_forward`
                with torch.no_grad():
                    try:
                        losses_step = - acq_func(X)  # (`batch_size`,) tensor
                    except (RuntimeError, NotPSDError) as e:
                        if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                            if e.args[0][:7] not in ['symeig_', 'cholesk']:
                                raise
                        losses_step = np.inf * torch.ones(self.num_restarts).to(**self.tkwargs)
                    loss: Tensor = losses_step.sum().detach()

                forw_results = acq_func.opt_forward(X, Y, eval_J=eval_J, new_samples=new_samples)

                g, f_Y = forw_results[:2]
                if eval_J:
                    losses_step: Tensor = - forw_results[-1]
                    with torch.no_grad():
                        loss: Tensor = losses_step.sum().detach()  # if J(x) = f(g(x)) has not been evaluated yet

                f_Y = - f_Y.sum()  # we want to maximize
                f_Y.backward()
                # loss = f_Y.detach()
                aux_kw = {'oracle_g': g,
                          'proj_X': lambda x_: columnwise_clamp_(x_, bounds[0], bounds[1])}
                if not isinstance(self.opt, ASCGD):
                    aux_kw['filter_inds'] = acq_func.z_filter

                    z_inds = torch.randint(0, acq_func.get_m(), size=(acq_func.K_g,))
                    z_filter = torch.zeros(acq_func.get_m(), dtype=bool).to(self.tkwargs['device'])
                    z_filter[z_inds] = 1
                    aux_kw['filter_inds_y_update'] = z_filter

                    aux_kw['oracle_y_g'] = lambda z: acq_func.oracle_g(z, custom_z_filter=z_filter)

                try:
                    optimizer.step(**_filter_kwargs(optimizer.step, **aux_kw))
                except (RuntimeError, NotPSDError) as e:
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    msg = f'({type(e)})'
                    break

                if isinstance(self.opt, ASCGD):
                    acq_func.set_z_ind_samples(z_filter, self.tkwargs['device'])
                    new_samples = False
            elif 'nested' in self.opt_name:
                try:
                    acq_func.gen_z_ind_samples(device=X.device)
                    hat_g_X = acq_func.oracle_g(X)
                    assert hat_g_X.shape == (self.num_restarts, q, acq_func.Kt_g), (
                        hat_g_X.shape, (self.num_restarts, q, acq_func.Kt_g))

                    loss = - acq_func.outer_f(hat_g_X).sum()
                    loss.backward()
                    optimizer.step()
                except (RuntimeError, NotPSDError) as e:
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    msg = f'({type(e)})'
                    break

            else:
                try:
                    # this performs batch evaluation, so this is an `batch_size`-dim tensor
                    losses_step: Tensor = - acq_func(X)  # shape: (t-batch,)
                except (RuntimeError, NotPSDError) as e:
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    msg = f'({type(e)})'
                    break
                loss = losses_step.sum()

                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            time_one_opt_step = time.time() - time_one_opt_step_ref

            if self.early_stop:
                stop = stopping_criterion.evaluate(fvals=loss.detach())

            # clamp values to the feasible set
            X.data = columnwise_clamp(X, bounds[0], bounds[1])
            # for j, (lb, ub) in enumerate(zip(*bounds)):
            #     X[..., j].clamp_(lb, ub)  # need to do this on the data not X itself
            if (i == 9 % 10 or stop or i == self.num_opt_steps - 1) and torch.all(torch.isfinite(X)):
                X_copy = X.detach().clone()

            if i % 10 == 0 and self.verbose > 1:
                print(f"Iteration {i + 1:>3}/{self.num_opt_steps:<3d}  |  Acquisition value: {-loss.item():>5.5f}")

            if stop:
                break
        X = X_copy
        if self.verbose > 0:
            print(
                f'Acquisition step {self.num_acq_steps:>3d}  |  '
                f'Last optimize step: {i + 1:>3d} / {self.num_opt_steps:<3d} in '
                f'{time.time() - self.acq_step_time_ref:>4.2f}s '
                f'({time.time() - self.acq_step_time_ref + self.total_ex_time:>4.2f}s)'
                f"{' ' + msg}"
            )
        # return only best among num_starts candidates
        with torch.no_grad():
            try:
                best_ind = torch.argmax(acq_func(X)).item()
            except (RuntimeError, NotPSDError) as e:
                if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                    if e.args[0][:7] not in ['symeig_', 'cholesk']:
                        print(e.args[0][:7])
                        raise
                print(f'Got {e} when trying to select best candidates among {self.num_restarts} candidates')
                best_ind = 0
                best_val = -np.inf
                for i, x in enumerate(X):
                    try:
                        val = acq_func(x.unsqueeze(0)).item()
                        if val > best_val:
                            best_val = val
                            best_ind = i
                    except Exception as ee:
                        print(i, ee)
                        pass
        X = X[:, :q]  # n_starts x q x d

        # loss obtained for the selected point at the last optimization step
        meta_dic = {'info': dict(last_loss=np.inf if losses_step is None else losses_step[best_ind].detach().item())}

        return X[best_ind], meta_dic

    def get_best(self) -> np.ndarray:
        """ Return the point at which the highest black-box value has been observed """
        return self.data_X.detach().cpu().numpy()[self.data_Y.detach().cpu().numpy().flatten().argmax()]

    def es_optimize_acqf_and_get_observation(self, acq_func, q, bounds) -> Tensor:
        """
        Maximise acquisition function with evolutionary algorithm

        Args:
            acq_func: acquistion function
            q: number of points to acquire
            bounds: acquisition function search space bounds

        Returns:
            new_X: `q x d` tensor of points to acquire
        """
        q_q_fantasies = q  # q + q_fantaisies when using Knowledge Gradient
        if isinstance(acq_func, OneShotAcquisitionFunction):
            q_q_fantasies += acq_func.num_fantasies
        try:
            X: Tensor = gen_batch_initial_conditions(acq_func, bounds, q_q_fantasies, self.num_restarts,
                                                     self.num_raw_samples,
                                                     options={'seed': self.seed + self.num_points})
        except (RuntimeError, NotPSDError) as e:
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print(f'{e.args[0][:13]} error handled during intitialization')
            X: Tensor = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(self.num_restarts, q_q_fantasies,
                                                                         self.input_dim, **self.tkwargs)

        assert X.shape == (self.num_restarts, q_q_fantasies, bounds.shape[
            -1]), f"X.shape should be {(self.num_restarts, q_q_fantasies, bounds.shape[-1])} but got {X.shape}"
        X: np.ndarray = X.cpu().numpy().reshape(self.num_restarts, -1)

        if 'nested' in self.opt_name:
            assert isinstance(acq_func, CompositionalAcquisition)
            kw = {'smooth': False}

            def acq_func_wr(x: Tensor):
                acq_func.gen_z_ind_samples(device=self.tkwargs['device'])
                return acq_func.nested_eval(x, **kw)
        else:
            acq_func_wr = acq_func
        opt_kw: Dict[str, Any] = dict(acq=acq_func_wr, bounds=bounds.cpu().numpy(),
                                      pop=self.num_restarts, q=q, iters=self.num_opt_steps, verbose=self.verbose - 1,
                                      tkwargs=self.tkwargs)

        if issubclass(self.opt, DEopt):
            opt_acq = DEopt(**opt_kw)

            new_X = opt_acq.optimize(initial_suggest=X, fix_input=None)
            assert new_X.shape == (q, self.input_dim)

        elif issubclass(self.opt, CMAESopt):

            opt_kw.update(**self.optimizer_kwargs)
            opt_acq = CMAESopt(**opt_kw)

            new_X = opt_acq.optimize(initial_suggest=X, fix_input=None)
            assert new_X.shape == (q, self.input_dim)
        else:
            raise ValueError(self.opt)
        new_X = torch.from_numpy(new_X).to(**self.tkwargs)
        return new_X

    def bfgs_optimize_acqf_and_get_observation(self, acq_func, q, bounds):
        """
        Maximise acquisition function with evolutionary L-BFGS-B method

        Args:
            acq_func: acquistion function
            q: number of points to acquire
            bounds: acquisition function search space bounds

        Returns:
            new_X: `q x d` tensor of points to acquire
        """
        q_q_fantasies = q  # q + q_fantaisies when using Knowledge Gradient
        options = {"maxiter": self.num_opt_steps, 'disp': self.verbose - 1, 'gtol': 1e-8, 'ftol': 1e-15}
        if isinstance(acq_func, OneShotAcquisitionFunction):
            q_q_fantasies += acq_func.num_fantasies
        try:
            X: Tensor = gen_batch_initial_conditions(acq_func, bounds, q_q_fantasies, self.num_restarts,
                                                     self.num_raw_samples,
                                                     options={'seed': self.seed + self.num_points})
        except (RuntimeError, NotPSDError) as e:
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print(f'{e.args[0][:13]} error handled during intitialization')
            X: Tensor = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(self.num_restarts, q_q_fantasies,
                                                                         self.input_dim, **self.tkwargs)

        assert X.shape == (self.num_restarts, q_q_fantasies, bounds.shape[
            -1]), f"X.shape should be {(self.num_restarts, q_q_fantasies, bounds.shape[-1])} but got {X.shape}"

        if isinstance(acq_func, CompositionalAcquisition):
            acq_func.approx = True
            acq_func.gen_z_ind_samples(device=X.device)
            fixed_features = None
            batch_limit: int = self.num_restarts
            batch_candidates_list: List[Tensor] = []
            batch_acq_values_list: List[Tensor] = []
            start_idcs = list(range(0, self.num_restarts, batch_limit))
            for start_idx in start_idcs:
                end_idx = min(start_idx + batch_limit, self.num_restarts)

                clamped_candidates = columnwise_clamp(
                    X=X[start_idx:end_idx], lower=bounds[0], upper=bounds[1]
                ).requires_grad_(True)

                shapeX = clamped_candidates.shape
                x0 = _arrayify(clamped_candidates.view(-1))
                scipy_bounds = make_scipy_bounds(
                    X=X, lower_bounds=bounds[0], upper_bounds=bounds[1]
                )
                constraints = []

                def f(x_):
                    X_ = (
                        torch.from_numpy(x_).to(X).view(shapeX).contiguous().requires_grad_(True)
                    )
                    trial = 0
                    done = False
                    X_fix = fix_features(X=X_, fixed_features=fixed_features)
                    while not done:
                        try:
                            loss = -acq_func.nested_eval(X_fix, smooth=True).sum()
                            done = True
                        except (RuntimeError, NotPSDError) as error:
                            if isinstance(error, RuntimeError) and not isinstance(error, NotPSDError):
                                if error.args[0][:7] not in ['symeig_', 'cholesk']:
                                    raise
                            trial += 1
                            print('new trial')
                            if trial >= 5:
                                loss = X_.mul(0).sum()  # setting grad to 0 will interupt optimization
                                done = True

                    # compute gradient w.r.t. the inputs (does not accumulate in leaves)
                    gradf = _arrayify(torch.autograd.grad(loss, X_)[0].contiguous().view(-1))
                    fval = loss.item()
                    return fval, gradf

                res = minimize(
                    f,
                    x0,
                    method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
                    jac=True,
                    bounds=scipy_bounds,
                    constraints=None,
                    callback=None,
                    options={k: v for k, v in options.items() if k != "method"},
                )
                candidates = fix_features(
                    X=torch.from_numpy(res.x).to(X).view(shapeX).contiguous(),
                    fixed_features=fixed_features,
                )
                batch_candidates_curr = columnwise_clamp(
                    X=candidates, lower=bounds[0], upper=bounds[1], raise_on_violation=True
                )
                with torch.no_grad():
                    try:
                        batch_acq_values_curr = acq_func.nested_eval(batch_candidates_curr, smooth=True)
                    except (RuntimeError, NotPSDError) as e:
                        if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                            if e.args[0][:7] not in ['symeig_', 'cholesk']:
                                print(e.args[0][:7])
                                raise
                        print(
                            f'Got {e} when tracq_func.nested_evalying to select best candidates among {self.num_restarts} candidates')
                        batch_acq_values_curr = torch.ones(batch_candidates_curr.shape[0]).mul(-np.inf)
                        for i, x in enumerate(batch_candidates_curr):
                            try:
                                batch_acq_values_curr[i] = acq_func.nested_eval(x.unsqueeze(0)).item()
                            except Exception as ee:
                                print(i, ee)
                                pass

                batch_candidates_list.append(batch_candidates_curr)
                batch_acq_values_list.append(batch_acq_values_curr)
            batch_candidates = torch.cat(batch_candidates_list)
            batch_acq_values = torch.cat(batch_acq_values_list)

            best = torch.argmax(batch_acq_values.view(-1), dim=0)
            new_X = batch_candidates[best]

            if isinstance(acq_func, OneShotAcquisitionFunction):
                new_X = acq_func.extract_candidates(X_full=new_X)

        else:
            new_X, batch_acq_values = optimize_acqf(
                batch_initial_conditions=X,
                acq_function=acq_func,
                bounds=bounds,
                q=q,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                return_best_only=True,
                options={"maxiter": self.num_opt_steps, 'disp': self.verbose - 1, 'gtol': 1e-8, 'ftol': 1e-15},
                **self.optimizer_kwargs)

        return new_X
