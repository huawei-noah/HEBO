from febo.utils import locate, get_logger
from types import FunctionType

from febo.environment.environment import EnvironmentConfig
from febo.utils.config import Configurable, EnumConfigField, ConfigField, assign_config, config_manager, \
    ClassListConfigField
from febo.environment.environment import Environment, ConstraintsMixin, NoiseObsMixin, NoiseObsMode
import numpy as np
import os
import json
from .noise import NoiseFunction, GaussianNoiseFunction
import yaml

logger = get_logger('environment')

class BenchmarkEnvironmentConfig(EnvironmentConfig):
    constraints = ClassListConfigField([])
    lower_bound_objective = ConfigField(None, field_type=float, allow_none=True)
    noise_function = ConfigField(0)
    noise_obs_mode = EnumConfigField('none', enum_cls=NoiseObsMode, comment='Can be set to "full", "evaluation" or "hidden".')
    dimension = ConfigField(3)
    num_domain_points = ConfigField(30)
    bias = ConfigField(0)
    scale = ConfigField(1)
    seed = ConfigField(None, comment='Seed for randomly generated environments.', allow_none=True)
    random_x0 = ConfigField(True)
    random_x0_min_value = ConfigField(None, allow_none=True)
    _section = 'environment.benchmark'

# config_manager.register(BenchmarkEnvironmentConfig)

@assign_config(BenchmarkEnvironmentConfig)
class BenchmarkEnvironment(NoiseObsMixin, ConstraintsMixin, Environment):
    """
    Base class for benchmark environments. Benchmark environment are implemented by specifying a single function _f(x)
    Further, benchmarks with known max_value should provide this in the `max_value` property.
    The base class also adds (safety) constraints and noise (TODO), depending on the configuration.
    """

    def __init__(self, path=None):
        super().__init__(path=path)
        self._x = None  # current parameter set in the environment
        self._max_value = None  # max value achievable
        self._s = self.config.constraints
        self._lower_bound_objective = self.config.lower_bound_objective
        self._num_constraints = len(self._s) + int(self._lower_bound_objective is not None)
        self._env_parameters = {}
        self._init_seed()
        self._x0 = None


    def initialize(self):
        self._init_noise_function()
        env_info = super().initialize()
        if self.config.noise_obs_mode == NoiseObsMode.full:
            env_info['noise_function'] = self._noise_function.std

        env_info['noise_obs_mode'] = self.config.noise_obs_mode

        if self.x0 is None:
            self._x0 = self.domain.l + self.domain.range/2

        if self.config.random_x0:
            logger.info("Using random initial point.")
            self._x0 = self._get_random_initial_point()

        env_info['x0'] = self.x0

        return env_info

    def _get_random_initial_point(self):
        x0 = self.domain.l + self.domain.range * np.random.uniform(size=self.domain.d)
        if not self.config.random_x0_min_value is None:
            while self.f(x0) < self.config.random_x0_min_value:
                x0 = self.domain.l + self.domain.range * np.random.uniform(size=self.domain.d)

            logger.info("Found initial feasible point.")
        return  x0

    def _get_noise_obs_fields(self):
        if self.config.noise_obs_mode in [NoiseObsMode.full, NoiseObsMode.evaluation]:
            return ["y", "s"]

        return []

    def _get_dtype_fields(self):
        return super()._get_dtype_fields() + [('y_exact', 'f8'), ('y_max', 'f8')]

    def evaluate(self, x=None):
        if x is not None:
            self._x = x

        evaluation = np.empty(shape=(), dtype=self.dtype)
        evaluation['x'] = self._x

        evaluation['y_exact'] = (self.f(self._x)).item()*self.config.scale + self.config.bias
        evaluation['y_max'] = self.max_value

        # if we use Gaussian Noise, we can query the std
        if isinstance(self._noise_function, GaussianNoiseFunction):
            # if noise is observed, add to evaluation
            if self.config.noise_obs_mode in [NoiseObsMode.evaluation, NoiseObsMode.full]:
                evaluation['y_std'] = np.asscalar(self._noise_function.std(self._x))

        evaluation['y'] = evaluation['y_exact'] + self._noise_function(self._x)
        for i, s in enumerate(self._s):
            evaluation['s'][i] = s(self._x) + self._noise_function(self._x)
            if isinstance(self._noise_function, GaussianNoiseFunction) and \
                    self.config.noise_obs_mode in [NoiseObsMode.evaluation, NoiseObsMode.full]:
                evaluation['s_std'][i] = np.asscalar(self._noise_function.std(self._x))
        if self._lower_bound_objective is not None:
            evaluation['s'][-1] = - (evaluation['y_exact'] - self.lower_bound_objective)

        return evaluation

    def f(self, x):
        """
        Function to be implemented by actual benchmark.
        Args:
            x:

        Returns:

        """
        raise NotImplementedError

    @property
    def x0(self):
        return self._x0

    def _init_noise_function(self):
        """"""
        if self.config.noise_function:

            # if self.config.noise_function is a number
            if isinstance(self.config.noise_function, (float, int)):
                noise_var = self.config.noise_function

                # define noise _function
                def noise_function(x):
                    x = np.atleast_2d(x)
                    return noise_var * np.ones(shape=(x.shape[0], 1))
            elif isinstance(self.config.noise_function, str):
                noise_function = locate(self.config.noise_function)
            else:
                raise Exception("Invalid setting for 'noise_function'.")
            # if noise_function is a function, assume it provides std for GaussianNoiseFunction
            if isinstance(noise_function, FunctionType):
                self._noise_function_cls = type(f"__{noise_function.__name__}_Noise", (GaussianNoiseFunction,),
                                            {'std': lambda self, x: noise_function(x)})
            elif issubclass(noise_function, NoiseFunction):
                self._noise_function_cls = noise_function
        else:
            self._noise_function_cls = NoiseFunction

        self._noise_function = self._noise_function_cls(self._domain, self.f)

    @property
    def max_value(self):
        if self._max_value is None:
            raise NotImplementedError

        return self._max_value*self.config.scale + self.config.bias

    @property
    def seed(self):
        """
        Provides a random seed for random generation of environments.
        """
        return self._seed

    @property
    def _requires_random_seed(self):
        """
        Overwrite this property and set to True to use random seed.
        """
        return False

    def _init_seed(self):
        """
        Initialize self.seed. First checks if self._path is provided, and if a file 'environment.yaml' exists in this path. If that file contains a dict {'seed', some_seed} this is used as seed. Else the the value from self.config.seed is taken, if this is None, a random integer is generated as seed. This seed, either randomly generated or from self.config.seed is saved in 'environment.yaml'.
        """
        # only if enviornment requires a random seed
        if not self._requires_random_seed:
            return

        # initialize seed
        self._seed = None

        # try to read seed from file
        if self._path:
            env_config_path = os.path.join(self._path, 'environment.yaml')
            if self._path and os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    data = yaml.load(f)
                    self._seed = data['seed']
                    logger.info("Using random seed from environment.yaml.")
        else:
            logger.warning('Path not provided, cannot load/save seed.')

        # if seed was not loaded from file
        if self._seed is None:
            self._seed = self.config.seed
            # no seed given in configuration, pick a random one
            if self._seed is None:
                logger.info("No random seed provided in config, choosing a random random seed.")
                self._seed = np.random.randint(2 ** 32 - 1)

            # save seed
            if self._path:
                env_config_path = os.path.join(self._path, 'environment.yaml')
                data = {'seed': self._seed}
                with open(env_config_path, 'w') as f:
                    yaml.dump(data, f)
                    logger.info("Saved random seed to environment.yaml.")
        elif self.config.seed is not None and self._seed != self.config.seed:
            logger.warning(
                "Seed from saved environment file is different than seed in config. Using seed from environment file.")

