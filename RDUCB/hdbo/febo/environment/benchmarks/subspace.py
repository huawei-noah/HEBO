from febo.environment import ContinuousDomain
from febo.environment.benchmarks import Camelback
from febo.environment.benchmarks.functions import Camelmod
from febo.environment.benchmarks.hpolib import Hartmann6
from febo.utils.config import Config, ConfigField, assign_config
import numpy as np

class AugmentedDimensionMixinConfig:
    aug_d = ConfigField(10)
    random_permutation = ConfigField(True)
    _section = 'environment.benchmark'

@assign_config(AugmentedDimensionMixinConfig)
class AugmentedDimensionMixin:

    def __init__(self, path=None):
        super().__init__(path)

        if not isinstance(self._domain, ContinuousDomain):
            raise Exception("Can only augment a ContinuousDomain!")
        self._orig_domain = self._domain

        l = np.concatenate((self._domain.l, -np.ones(self.config.aug_d)))
        u = np.concatenate((self._domain.u, np.ones(self.config.aug_d)))
        total_d = self._domain.d + self.config.aug_d

        # define permutation
        self._per = np.arange(total_d)
        if self.config.random_permutation:
            self._per = np.random.permutation(total_d)
        # compute inverse permutation
        self._inv_per = np.argsort(self._per)

        self._domain = ContinuousDomain(l[self._per],u[self._per])
        self._x0 = np.concatenate((self._x0, np.zeros(self.config.aug_d)))

    def f(self, X):
        X = X[self._inv_per]  # undo permutation
        X = X[:self._orig_domain.d]  # take active dimensions
        return super(AugmentedDimensionMixin, self).f(X)  # return original function


class AugmentedCamelback(AugmentedDimensionMixin, Camelback):
    pass

class AugmentedCamelmod(AugmentedDimensionMixin, Camelmod):
    pass


class AugmentedHartmann6(AugmentedDimensionMixin, Hartmann6):
    pass