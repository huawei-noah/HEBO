from febo.environment import ContinuousDomain
from febo.utils import get_logger
from febo.utils.config import assign_config, ConfigField, Configurable, Config, config_manager
from febo.environment.domain import DiscreteDomain
import numpy as np

logger = get_logger('environment')

class NoiseConfig(Config):
    low = ConfigField(0.5, comment="May be used by the noise function to roughly set the lowest noise level.")
    high = ConfigField(0.5, comment="May be used by the noise function to roughly set the higest noise level.")
    seed = ConfigField(None, comment="Seed for randomly generated noise function.", allow_none=True)
    _section = 'environment.benchmark.noise'

# config_manager.register(NoiseConfig)

@assign_config(NoiseConfig)
class NoiseFunction(Configurable):
    """
    Base class to implement custom noise functions.
    Note that the configuration of BenchmarkEnvironment.noise_function also takes an ordinary python function.

    Args:
        x: evaluation point

    Returns:
        Variance at x

    """
    def __init__(self, domain, benchmark_f=None):
        self._domain = domain
        self._f = benchmark_f

        if not isinstance(domain, self.allowed_domains):
            raise Exception(f"Invalid domain {domain.__class__.__name__} for this noise function.")

        if self.config.high < self.config.low:
            logger.error("Config environment.benchmark.noise:high high is smaller than environment.benchmark.noise:low")


    def __call__(self, X=None):
        return 0.0

    @property
    def allowed_domains(self):
        """
        Returns: tuple of allowed domain classes.

        """
        return (DiscreteDomain, ContinuousDomain)


class GaussianNoiseFunction(NoiseFunction):

    def std(self, X=None):
        raise NotImplementedError

    def __call__(self, X=None):
        return np.random.normal(scale=self.std(X))


class ExpPropValueNoise(GaussianNoiseFunction):
    def std(self, X=None, Y=None):
        return np.exp(Y) + self.config.low

class ReluValueNoise(GaussianNoiseFunction):

    def std(self, X=None):
        Y = self._f(X)
        return np.maximum(Y*self.config.high, self.config.low)

class ReluInvValueNoise(GaussianNoiseFunction):

    def std(self, X=None):
        Y = 1 - self._f(X)
        return np.minimum(self.config.high, np.maximum(Y*self.config.high, self.config.low))

class TanHPropValueNoise(GaussianNoiseFunction):

    def std(self, X=None):
        Y = self._f(X)
        return self.config.high*(np.tanh(8*(Y-0.6)) + 1)/2 + self.config.low

class TanHInvPropValueNoise(TanHPropValueNoise):
    def std(self, X=None):
        Y = self._f(X)
        return self.config.high*(2 - np.tanh(8*(Y-0.6)))/2 + self.config.low

class ContinuousNoiseFunction(GaussianNoiseFunction):

    @property
    def allowed_domains(self):
        return (ContinuousDomain,)

    def std(self, X=None, Y=None):
        return self.n(X)

    def n(self, X):
        """
        Implementation of noise function. X is normalized to unit cube.
        Args:
            X:

        Returns: Noise value at X

        """
        raise NotImplementedError

class HighNoiseAroundOrigin(ContinuousNoiseFunction):

    def n(self, X):
        return (self.config.high-self.config.low)* np.exp(-10 * np.sum((X - 0.5) ** 2)) + self.config.low

class LowNoiseAroundOrigin(ContinuousNoiseFunction):

    def n(self, X):
        return (self.config.high-self.config.low)* (1-np.exp(-10 * (X - 0.5) **2)) + self.config.low

class Sin1D(GaussianNoiseFunction):

    def std(self, X):
        return (-np.sin(50*np.pi*X) + 1)/2*(self.config.high -self.config.low) + self.config.low

class SinNorm(ContinuousNoiseFunction):

    def n(self, X):
        X = np.atleast_2d(X)
        X = np.linalg.norm(X, axis=1).reshape(-1, 1)
        return (-np.sin(2*np.pi*X) + 1)/2*(self.config.high -self.config.low) + self.config.low

class RandomNoiseConfig(NoiseConfig):
    _section = 'environments.benchmark.noise.random'


class RandomNoise(GaussianNoiseFunction):
    def __init__(self, domain):
        super().__init__(domain)

        if not len(np.unique(self._domain.points, axis=0)) == len(self._domain.points):
            raise Exception("Need unique domain points to add random noise.")

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self._rho = np.random.uniform(self.config.low, self.config.high,
                                      size=self._domain.num_points).reshape(-1,1) # generate uniform noise bound

    @property
    def allowed_domains(self):
        return (DiscreteDomain,)

    def std(self, x, y=None):
        x = np.atleast_2d(x)
        return np.vstack([self._rho[(row == self._domain.points).all(axis=1)] for row in x])