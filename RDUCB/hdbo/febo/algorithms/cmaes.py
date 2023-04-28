from febo.algorithms import Algorithm, AlgorithmConfig
from cma import CMAEvolutionStrategy
import numpy as np

from febo.utils.config import ConfigField, assign_config


class CMAESConfig(AlgorithmConfig):
    sigma0 = ConfigField(0.1)
    _section = 'algorithm.cmaes'

@assign_config(CMAESConfig)
class CMAES(Algorithm):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        if self.x0 is None:
            self.x0 = self.domain.l + self.domain.range/2

        # cma operates on normalized scale
        x0 = self.domain.normalize(self.x0)
        self.cma = CMAEvolutionStrategy(x0=x0, sigma0=self.config.sigma0,  inopts={'bounds': [0,1]})
        self._X = None
        self._X_i = 0
        self._Y = None

    def _next(self, context=None):
        if self._X is None:
            # get new population
            self._X = self.cma.ask()
            self._Y = np.empty(len(self._X))
            self._X_i = 0

        return self.domain.denormalize(self._X[self._X_i])

    def finalize(self):
        self.cma.result_pretty()


    def best_predicted(self):
        xbest = None
        if self.cma.result.xbest is not None:
            xbest = self.domain.denormalize(self.cma.result.xbest)

        return xbest if not xbest is None else self.x0


    def add_data(self, data):
        self._Y[self._X_i] = data['y']
        self._X_i += 1

        # population complete
        if self._X_i == len(self._X):
            self.cma.tell(self._X, -self._Y)
            self._X = None

        super().add_data(data)
