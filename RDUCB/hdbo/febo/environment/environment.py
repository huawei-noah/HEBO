import numpy as np
from functools import wraps
from enum import Enum

from febo.utils import get_logger
from febo.utils.config import ConfigField, Config, Configurable, assign_config, config_manager


class EnvironmentConfig(Config):
    _section = 'environment'

# config_manager.register(EnvironmentConfig)

logger = get_logger("environment")

@assign_config(EnvironmentConfig)
class Environment(Configurable):
    def __init__(self, path=None):
        self._domain = None
        self._tmax = None
        self._path = path

    @property
    def name(self):
        return f"{type(self).__module__}.{type(self).__name__}"

    @property
    def domain(self):
        return self._domain

    def initialize(self):
        """
        Initialize domain.

        Returns (dict):
            Dictionary containing environment information, which is passed as kwargs to the algorithm.
            By default, { 'domain' : self.domain } is returned.

        """
        return { 'domain': self.domain }

    def evaluate(self, x=None):
        raise NotImplementedError

    def finalize(self):
        pass

    def _get_dtype_fields(self):
        """

        :return:
        """
        return [('x', f'({self.domain.d},)f8'), ('y', 'f8')]

    @property
    def dtype(self):
        return np.dtype(self._get_dtype_fields())

    @property
    def Tmax(self):
        return self._tmax


class NoiseObsMode(Enum):
    none = 1
    evaluation = 2
    full = 3

class NoiseObsMixin:
    """
    Environment Mixin that adds noise observations
    """
    def __init__(self, path=None):
        super(NoiseObsMixin, self).__init__(path=path)

    def _get_noise_obs_fields(self):
        """
        List of fields (strings) where noise is observed. By default ["y"].
        Returns:

        """
        return ["y"]

    def _get_dtype_fields(self):
        fields = super(NoiseObsMixin, self)._get_dtype_fields()

        noise_obs_fields = self._get_noise_obs_fields()
        new_fields = []
        for field, dtype in fields:
            if field in noise_obs_fields:
                new_fields.append( (f"{field}_std", dtype) )

        return fields + new_fields

    @property
    def noise_obs_mode(self):
        raise NotImplementedError


class ConstraintsMixin:
    def __init__(self,path=None):
        super(ConstraintsMixin, self).__init__(path=path)
        self._num_constraints = 0
        self._lower_bound_objective = None

    def initialize(self):
        info = super(ConstraintsMixin, self).initialize()
        info['num_constraints'] = self._num_constraints
        info['lower_bound_objective'] = self._lower_bound_objective
        return info

    @property
    def num_constraints(self):
        return self._num_constraints

    @property
    def lower_bound_objective(self):
        return self._lower_bound_objective

    def _get_dtype_fields(self):
        fields = super(ConstraintsMixin, self)._get_dtype_fields()

        if self._num_constraints:
            fields += [('s', f"({self._num_constraints},)f8")]

        return fields

class ContextMixin:
    """
    TODO
    """

    def __init__(self, *args, **kwargs):
        super(ContextMixin, self).__init__(*args, **kwargs)

    def get_context(self):
        raise NotImplementedError


