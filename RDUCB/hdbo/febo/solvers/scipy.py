from multiprocessing.pool import Pool
import numpy as np

from febo.solvers.multi_lbfgs import _minimize_lbfgsb_multi
from .solver import Solver
from scipy.optimize import minimize as scipy_minimize

from .seeds import Seeds
from febo.utils.config import ConfigField, Config, assign_config, config_manager
from febo.utils import get_logger, split_int


class ScipySolverConfig(Config):
    lbfgs_use_gradients = ConfigField(False)
    lbfgs_maxfun = ConfigField(1000)
    # lbfgs_maxiter = ConfigField(1000)
    num_restart = ConfigField(50)
    num_processes = ConfigField(1)
    sync_restarts = ConfigField(True)
    convergence_warnings = ConfigField(True)
    _section = 'solver.scipy'

# config_manager.register(ScipyOptimizerConfig)

logger = get_logger('solver.scipy')

class FunWrapper:
    def __init__(self, fun, use_gradients, safe_mode=False, safety_wrapper=None):
        self._fun = fun
        self.best_x = None
        self.best_y = 10e10
        self._use_gradients = use_gradients
        self._safe_mode = safe_mode
        self._safety_wrapper = safety_wrapper

    def __call__(self, x):
        x = np.atleast_2d(x)
        res = self._fun(x)
        if self._use_gradients:
            y, grad = res
        else:
            y = res

        safe = [True] * len(x)
        if self._safe_mode:
            y, safe = self._safety_wrapper(x, y)  # pass through safety wrapper

        for xx, yy, ss in zip(x,y, safe):
            if yy < self.best_y and ss:
                self.best_y = yy
                self.best_x = xx.copy()

        if self._use_gradients:
            return y, grad
        else:
            return y

@assign_config(ScipySolverConfig)
class ScipySolver(Solver):

    def __init__(self, *args, **kwargs):
        super(ScipySolver, self).__init__(*args, **kwargs)
        self.seeds = Seeds(self._domain, self.initial_x)


        if self.config.num_processes > 1:
            pool = Pool(self.config.num_processes)
            self.map = pool.map
        else:
            self.map = map

        self._bounds = [(l,u) for l,u in zip(self._domain.l, self._domain.u)]
        self.safe_mode = False
        self.safety_wrapper = None
        self.infeasible_exception = True

    @property
    def requires_gradients(self):
        return self.config.lbfgs_use_gradients

    def minimize(self, f):
        self.seeds.new_iteration()
        fun = FunWrapper(f, self.requires_gradients, self.safe_mode, self.safety_wrapper)
        restarts_per_process = split_int(self.config.num_restart, self.config.num_processes)
        opt = {'maxfun': self.config.lbfgs_maxfun }
        restart_args = [{
            'fun':fun,
             'X0':np.array([self.seeds.next() for _ in range(restarts_per_process[p])]),
            'use_gradients' : self.requires_gradients,
            'bounds': self._bounds,
            'sync_restarts': self.config.sync_restarts,
            'warnings' : self.config.convergence_warnings,
            'options': opt}
                for p in range(self.config.num_processes)]
        res = self.map(minimize, restart_args)


        best_y = 10e10
        best_x = None
        for x,y in res:
            if y < best_y:
                best_y = y
                best_x = x

        if best_x is None and self.infeasible_exception:
            raise Exception("Optimizer did not find a feasible point!")

        # adding best x to seeds
        if best_x is not None:
            self.seeds.add_to_tail(best_x)

        return best_x, best_y




class SafeScipyOptimizer(ScipySolver):

    def _wrap_f_grad(self, x):
        y, grad, safe = self.f(x)

        if safe and y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()

        return y, grad

    def _wrap_f(self, x):
        y, safe = self.f(x)

        if safe and y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()

        return y

    def _initialize_seeds(self):
        sfun = self._s_grad if self.requires_gradients else self._s
        self.seeds.new_iteration(sfun = sfun)

    def _s_grad(self, x):
        y, grad, safe = self.f(x)
        return safe

    def _s(self, x):
        y, safe = self.f(x)
        return safe


def minimize(args):
    """
    Wrapper around scipy.optimizer.minimize, which can be called using multi processing.
    This method does multiple restarts for each x0 in X0 provided.

    """
    fun = args['fun']
    X0 = args['X0']
    use_gradients = args['use_gradients']
    bounds = args['bounds']
    sync_restarts = args['sync_restarts']
    warnings = args['warnings']
    options = args['options']

    warnings = {} # dict to collect warnings
    if sync_restarts:
         res = scipy_minimize(fun, X0, method=_minimize_lbfgsb_multi, jac=use_gradients, tol=0.001, bounds=bounds, options = options)
         for status,mes in zip(res['status'], res['message']):

             if status:
                 if not mes in warnings:
                     warnings[mes] = 0
                 warnings[mes] += 1
    else:
        for x in X0:
            res = scipy_minimize(fun, x, method="L-BFGS-B", jac=use_gradients, tol=1e-7, bounds=bounds, options = options)
            if res['status']:
                mes = res['message']

                if not mes in warnings:
                    warnings[mes] = 0
                warnings[mes] += 1

    if warnings:
        for mes, num in warnings.items():
            logger.warning(f'Optimizer Warning ({num}x): {mes}')
    return fun.best_x, fun.best_y
