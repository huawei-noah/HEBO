import numpy as np
from febo.algorithms.safety import SafetyMixin

from febo.environment import ContinuousDomain
from febo.utils import locate

from febo.environment.domain import DiscreteDomain
# from models.model import ScaleShiftModel
from febo.utils import get_logger
from febo.utils.config import ConfigField, Config, Configurable, assign_config, config_manager, ClassConfigField
import time



logger = get_logger("algorithm")

class AlgorithmConfig(Config):
    _section = 'algorithm'

# config_manager.register(AlgorithmConfig)

@assign_config(AlgorithmConfig)
class Algorithm(Configurable):
    """
    Base class for algorithms.
    """

    def __init__(self, **experiment_info):
        self.experiment_info = experiment_info
        self._dtype_fields = []
        self.name = type(self).__name__

    def initialize(self, **kwargs):
        """
        Called to initialize the algorithm. Resets the algorithm, discards previous data.

        Args:
            **kwargs: Arbitrary keyword arguments, recieved from environment.initialize().
            domain (DiscreteDomain, ContinousDomain): Mandatory argument. Domain of the environment
            initial_evaluation: (Optional) Initial evaluation taken from the environment


        Returns:

        """
        self.domain = kwargs.get("domain")
        self.x0 = kwargs.get("x0", None)
        self.initial_data = kwargs.get("initial_data", [])
        logger.info(f"Got {len(self.initial_data)} initial data points.")
        self._exit = False
        self.t = 0

        # TODO Move the next two arguments into a separate class?
        self.lower_bound_objective_value = kwargs.get("lower_bound_objective_value", None)
        self.num_constraints = kwargs.get("num_constraints", None)
        self.noise_obs_mode = kwargs.get("noise_obs_mode", None)

        self.__best_x = None
        self.__best_y = -10e10

    def _next(self, context=None):
        """
        Called by next(), used to get proposed parameter.
        Opposed to ``next()``, does return only x, not a tuple.
        Returns: parameter x

        """
        raise NotImplementedError

    def next(self, context=None):
        """
        Called to get next evaluation point from the algorithm.
        By default uses  self._next() to get a proposed parameter, and creates additional_data

        Returns: Tuple (x, additional_data), where x is the proposed parameter, and additional_data is np 1-dim array of dtype self.dtype

        """
        if context is None:
            # call without context (algorithm might not allow context argument)
            next_x = self._next()
        else:
            # call with context
            next_x = self._next(context)

        if isinstance(next_x, tuple):
            x = next_x[0]
            additional_data = next_x[1]
        else:
            x = next_x
            additional_data = {}
        additional_data['t'] = self.t
        self.t += 1

        # for continous domains, check if x is inside box
        if isinstance(self.domain, ContinuousDomain):
            if (x > self.domain.u).any() or (x < self.domain.l).any():
                # logger.warning(f'Point outside domain. Projecting back into box.\nx is {x}, with limits {self.domain.l}, {self.domain.u}')
                x = np.maximum(np.minimum(x, self.domain.u), self.domain.l)

        return x, additional_data

    def add_data(self, data):
        """
        Add observation data to the algorithm.

        Args:
            data: TBD

        """
        if data['y'] > self.__best_y:
            self.__best_y = data['y']
            self.__best_x = data['x']

        self.initial_data.append(data)

    @property
    def dtype(self):
        """
        Returns:
            Numpy dtype of additional data return with next().

        """
        return np.dtype(self._get_dtype_fields())

    def _get_dtype_fields(self):
        """
        Fields used to define ``self.dtype``.

        Returns:

        """
        fields = [("t", "i")]
        return fields

    def finalize(self):
        return {'initial_data' : self.initial_data,
                'best_x' : self.best_predicted()}

    @property
    def requires_x0(self):
        """
        If true, algorithm requires initial evaluation from environment.
        By default set to False.
        """
        return False

    @property
    def exit(self):
        return self._exit

    def best_predicted(self):
        """
        If implemented, this should returns a point in the domain, which is currently believed to be best
        Returns:

        """
        return self.__best_x


