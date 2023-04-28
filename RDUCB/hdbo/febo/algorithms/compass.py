from febo.algorithms import AlgorithmConfig
from febo.algorithms.thread import ThreadAlgorithm
from febo.utils.config import ConfigField, assign_config
from noisyopt import minimizeCompass

class CompassConfig(AlgorithmConfig):
    deltatol = ConfigField(0.01)
    deltainit = ConfigField(0.5)
    redfactor = ConfigField(1.5)
    niter = ConfigField(400)
    _section = 'algorithm.cmaes'

@assign_config(CompassConfig)
class Compass(ThreadAlgorithm):

    def initialize(self, **kwargs):

        super().initialize(**kwargs)

    def minimize(self):
        # Compass requires an initial point
        if self.x0 is None:
            self.x0 = self.domain.l + self.domain._range / 2

        res = minimizeCompass(self.f, bounds=self.domain.bounds, x0=self.x0,
                              niter=self.config.niter,
                              deltatol=self.config.deltatol,
                              deltainit=self.config.deltainit,
                              scaling=self.domain.range,
                              redfactor=self.config.redfactor,
                              errorcontrol=False,
                              funcNinit=30,
                              alpha=0.05,
                              paired=False)
        x0 = res['x']
        res = minimizeCompass(self.f, bounds=self.domain.bounds, x0=x0,
                              niter=self.config.niter,
                              deltatol=self.config.deltatol,
                              deltainit=2*self.config.deltatol,
                              scaling=self.domain.range,
                              redfactor=self.config.redfactor,
                              errorcontrol=True,
                              funcNinit=30,
                              alpha=0.05,
                              paired=False)

    def best_predicted(self):
        return self._x