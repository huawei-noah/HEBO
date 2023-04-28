import safeopt
import numpy as np
from febo.algorithms import Algorithm, ModelMixin, AlgorithmConfig
from febo.utils import get_logger
from febo.utils.config import assign_config, ConfigField
import GPy

logger = get_logger('algorithm')

class SafeOptConfig(AlgorithmConfig):
    ucb = ConfigField(False)
    points_per_dimension = ConfigField(100)
    _section = 'algorithm.safeopt'

class SafeOptSwarmMod(safeopt.SafeOptSwarm):
    def __init__(self, *args, **kwargs):
        self.x0 = kwargs.pop('x0')
        super().__init__(*args, **kwargs)

    def optimize(self, ucb=False):
        # Make sure the safe set is still safe
        try:
            return super().optimize(ucb=ucb)
        except RuntimeError:
            logger.warning("Empty safeset, choosing initial point")
            return self.x0

        return super().optimize(ucb=ucb)

class SafeOptStub:
    """ just to ease evaluation of first data point. """
    def __init__(self, x0):
        self.x0 = x0

    def optimize(self, ucb=False):
        return self.x0

    def get_maximum(self):
        return self.x0, None


@assign_config(SafeOptConfig)
class SafeOpt(ModelMixin, Algorithm):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self._lower_bound_objective = kwargs['lower_bound_objective']
        self.safeopt = SafeOptStub(self.x0)

    def _next(self):
        try:
            return self.safeopt.optimize(ucb=self.config.ucb)
        except EnvironmentError:
            logger.warning("Empty safeset, choosing initial point")
            return self.x0

    def add_data(self, data):
        super().add_data(data)
        if isinstance(self.safeopt, SafeOptStub):
            self._initialize_safeopt(data['x'], data['y'])
        else:
            self.safeopt.add_new_data_point(data['x'], data['y'])

    def best_predicted(self):
        res = self.safeopt.get_maximum()
        if res is None:
            return self.x0
        else:
            return res[0]

    def _initialize_safeopt(self, x0, y0):
        kernel = self.model._get_kernel()
        gp = GPy.models.GPRegression(x0.reshape(1,-1), y0.reshape(1,1), kernel, noise_var=self.model.config.noise_var)
        bounds = [(l,u) for l,u in zip(self.domain.l, self.domain.u)]
        parameter_set = safeopt.linearly_spaced_combinations(bounds, self.config.points_per_dimension)
        self.safeopt = safeopt.SafeOpt(gp, parameter_set, self._lower_bound_objective, threshold=self._lower_bound_objective)


@assign_config(SafeOptConfig)
class SwarmSafeOpt(SafeOpt):

    def _initialize_safeopt(self, x0, y0):
        kernel = self.model._get_kernel()
        gp = GPy.models.GPRegression(x0.reshape(1, -1), y0.reshape(1, 1), kernel, noise_var=self.model.config.noise_var)
        bounds = [(l, u) for l, u in zip(self.domain.l, self.domain.u)]
        self.safeopt = SafeOptSwarmMod(gp, self._lower_bound_objective, bounds=bounds,
                                       threshold=self._lower_bound_objective, x0=x0)

