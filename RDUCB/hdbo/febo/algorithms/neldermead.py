from febo.algorithms.thread import ThreadAlgorithm
from febo.algorithms import AlgorithmConfig
from scipy.optimize import minimize
import numpy as np
from febo.utils.utils import get_logger
from febo.utils.config import ConfigField, assign_config

logger = get_logger('algorithm.nelder_mead')

class NelderMeadConfig(AlgorithmConfig):
    contraction_factor = ConfigField(0.3)
    initial_stepsize = ConfigField(0.1)
    restart_threshold = ConfigField(0.001)
    adaptive = ConfigField(True)
    _section = 'algorithm.nelder_mead'

@assign_config(NelderMeadConfig)
class NelderMead(ThreadAlgorithm):

    def initialize(self, **kwargs):
        self._bext_x_nelder_mead = None

        super().initialize(**kwargs)


    def f(self, x):
        # provide normalized access
        return super().f(self.domain.project(self.denormalize(x)))

    def normalize(self, x):
        return (x - self.domain.l)/self.domain.range

    def denormalize(self, x):
        return x*self.domain.range + self.domain.l

    def minimize(self):
        # NelderMead requires an initial point
        if self.x0 is None:
            self.x0 = self.domain.l + self.domain._range / 2

        # normalize x0
        self._x0 = (self.x0 - self.domain.l)/self.domain.range
        self._stepsize = self.config.initial_stepsize

        # keep restarting Nelder-Mead after convergence
        while True:
            # initial_simplex = self._stepsize *np.eye(self.domain.d) + self._x0

            # choose d initial points at random around x0
            initial_simplex = self._x0 + self._stepsize * np.random.uniform(size=self.domain.d ** 2).reshape(-1, self.domain.d)
            initial_simplex = np.vstack((initial_simplex, self._x0))
            # make sure initial_simplex is in domain
            initial_simplex = np.maximum(np.minimum(initial_simplex, 1), 0)

            # options:
            # set fatol to large value,
            # to ignore it as stopping condition (because of the noise)
            # x0 is overwritten by 'initial_simplex'
            xatol = self.config.restart_threshold
            res = minimize(self.f, x0=self._x0, method='Nelder-Mead', options={'maxiter' : 10000,
                                                                    'maxfev' : 10000,
                                                                    'initial_simplex' : initial_simplex,
                                                                    'adaptive' : self.config.adaptive,
                                                                    'xatol': xatol,
                                                                    'fatol' : 10e12})

            self._x0 = res['x']
            self._bext_x_nelder_mead = self.domain.project(self.denormalize(res['x'].copy()))
            logger.info(f"Saved best_x")

            self._stepsize *= self.config.contraction_factor
            self._stepsize = max(0.1, self._stepsize)
            logger.info(f"Restarting nelder-mead at {self.denormalize(self._x0)}, contraction factor {self.config.contraction_factor}")

    def best_predicted(self):
        if not self._bext_x_nelder_mead is None:
            return self._bext_x_nelder_mead

        return super().best_predicted()
