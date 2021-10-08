from typing import Dict

import numpy  as np
import torch
from botorch.utils import draw_sobol_samples
from gpytorch.utils.errors import NotPSDError
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.so_de import DE
from pymoo.configuration import Configuration
from pymoo.model.problem import get_problem_from_func
from pymoo.optimize import minimize

Configuration.show_compile_hint = False


class EvolutionOpt:
    def __init__(self,
                 acq, bounds: np.ndarray, q: int, tkwargs: Dict = {}, save_loss: bool = False,
                 pop: int = 100, iters: int = 500, verbose: float = 0):
        self.acq = acq
        self.pop = pop
        self.iter = iters
        self.tkwargs = tkwargs
        self.verbose = verbose
        self.q = q
        self.lb = np.repeat(bounds[0].reshape(1, -1), self.q, axis=0).flatten()
        self.ub = np.repeat(bounds[1].reshape(1, -1), self.q, axis=0).flatten()
        self.tensor_bounds = torch.tensor(bounds).to(**tkwargs)
        self.best_acq = []
        self.save_loss = save_loss
        self.first_eval = True

    def callback(self):
        pass

    def pymoo_obj(self, x: np.array, out: dict, *args, **kwargs):
        num_x = x.shape[0]
        assert x.ndim == 2, x.shape
        x = x.reshape(x.shape[0], self.q, -1)
        x = torch.from_numpy(x).to(**self.tkwargs)
        if 'init' in kwargs and kwargs['init']:
            x = x[:,:1]  # if init, the value of the acquisision function won't be used, only the output format

        with torch.no_grad():
            n_trial = 0
            n_trial_limit = 5
            success = False
            while n_trial < n_trial_limit and not success:
                try:
                    acq_eval = self.acq(x).cpu().numpy().reshape(num_x, 1)
                    success = True
                except RuntimeError as e:
                    n_trial += 1
                    print(f'Got error {e.args[0][:20]}, trial n. {n_trial}')
                    if n_trial == n_trial_limit:
                        raise e
            out['F'] = - acq_eval

        if self.save_loss:
            if not self.first_eval:
                self.best_acq.append(out['F'].min())
            else:
                self.first_eval = False

        self.callback()


class DEopt(EvolutionOpt):

    def __init__(self, acq, bounds: np.ndarray, q: int, tkwargs: Dict = {}, save_loss: bool = False,
                 pop: int = 100, iters: int = 500, verbose: float = 0, **kwargs):
        super().__init__(acq=acq, bounds=bounds, q=q, tkwargs=tkwargs, save_loss=save_loss, pop=pop, iters=iters,
                         verbose=verbose)

    def optimize(self, initial_suggest: np.ndarray = None, fix_input: dict = None) -> np.ndarray:
        self.fix = fix_input
        if initial_suggest is not None:
            n_rand = self.pop - initial_suggest.shape[0]
            if n_rand == 0:
                init_pop = initial_suggest
            else:
                init_pop = draw_sobol_samples(self.tensor_bounds, n=n_rand, q=self.q).numpy().reshape(n_rand, -1)
                init_pop = np.vstack([initial_suggest, init_pop])
        else:
            init_pop = draw_sobol_samples(self.tensor_bounds, n=self.pop, q=self.q).numpy().reshape(self.pop, -1)
        algo = DE(pop_size=self.pop, dither='vector', sampling=init_pop)
        prob = get_problem_from_func(self.pymoo_obj, xl=self.lb, xu=self.ub, n_var=len(self.lb), func_args={'init': True})
        try:
            res = minimize(prob, algo, ('n_gen', self.iter), verbose=self.verbose > 0)
            if res.X is None:  # no feasible solution founc
                opt_x = np.array([ind.X for ind in res.pop])
            else:
                opt_x = res.X.reshape(-1, len(self.lb))
        except (RuntimeError, NotPSDError) as e:
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            opt_x = init_pop[0]

        if self.fix is not None:
            for k, v in self.fix.items():
                opt_x[k] = v
        return opt_x.reshape(self.q, -1)


class CMAESopt(EvolutionOpt):

    def __init__(self, acq, bounds: np.ndarray, q: int, tkwargs: Dict = {}, save_loss: bool = False,
                 pop: int = 100, iters: int = 500, sigma: float = .25, tolfun: float = 1e-6, tolx: float = 1e-6,
                 verbose: float = 0, bipop: bool = False):
        super().__init__(acq, bounds, q, tkwargs, save_loss, pop, iters, verbose)
        self.sigma = sigma
        self.bipop = bipop
        self.tolfun = tolfun
        self.tolx = tolx

    def optimize(self, initial_suggest: np.ndarray = None, fix_input: dict = None) -> np.ndarray:
        self.fix = fix_input
        if initial_suggest is not None:
            n_rand = self.pop - initial_suggest.shape[0]
            if n_rand == 0:
                init_pop = initial_suggest
            else:
                init_pop = draw_sobol_samples(self.tensor_bounds, n=n_rand, q=self.q).numpy().reshape(n_rand, -1)
                init_pop = np.vstack([initial_suggest, init_pop])
        else:
            init_pop = draw_sobol_samples(self.tensor_bounds, n=self.pop, q=self.q).numpy().reshape(self.pop, -1)
        algo = CMAES(x0=init_pop, sigma=self.sigma, parallelize=True, maxfevals=self.iter * self.pop, restarts=np.inf,
                     incpopsize=1, popsize=self.pop, tolfun=self.tolfun, bipop=self.bipop,
                     tolx=self.tolx)

        prob = get_problem_from_func(self.pymoo_obj, xl=self.lb, xu=self.ub, n_var=len(self.lb), func_args={'init': True})
        try:
            res = minimize(prob, algo, ('n_gen', self.iter), verbose=self.verbose > 0)
            if res.X is None:  # no feasible solution founc
                opt_x = np.array([ind.X for ind in res.pop])
            else:
                opt_x = res.X.reshape(-1, len(self.lb))
        except (RuntimeError, NotPSDError) as e:
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            opt_x = init_pop[0]

        if self.fix is not None:
            for k, v in self.fix.items():
                opt_x[k] = v

        return opt_x.reshape(self.q, -1)
