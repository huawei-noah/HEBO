from febo.algorithms import AlgorithmConfig
from febo.algorithms.thread import ThreadAlgorithm
from noisyopt import minimizeSPSA

from febo.utils.config import ConfigField, assign_config


class SPSAConfig(AlgorithmConfig):
    a = ConfigField(0.5)
    c = ConfigField(0.1)
    niter = ConfigField(500)

    _section = 'algorithm.spsa'

@assign_config(SPSAConfig)
class SPSA(ThreadAlgorithm):

    def initialize(self, **kwargs):

        super().initialize(**kwargs)

    def minimize(self):
        # NelderMead requires an initial point
        if self.x0 is None:
            self.x0 = self.domain.l + self.domain._range / 2

        res = minimizeSPSA(self.f, bounds=self.domain.bounds, x0=self.x0, niter=self.config.niter, paired=False,
                           a=self.config.a, c=self.config.c)

    def best_predicted(self):
        return self._x