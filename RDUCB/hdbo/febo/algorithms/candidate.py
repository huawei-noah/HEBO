from febo.algorithms.algorithm import AlgorithmConfig
from febo.utils import cartesian

from .algorithm import Algorithm
import numpy as np
from febo.utils.config import Config, ConfigField, assign_config, config_manager, ClassConfigField


class CandidateAlgorithmConfig(AlgorithmConfig):
    candidates = ClassConfigField(None, comment="Function which returns a list of candidate points to be evaluated.", allow_none=True)
    _section = "algorithm.candidate"

# config_manager.register(CandidateAlgorithmConfig)


@assign_config(CandidateAlgorithmConfig)
class CandidateAlgorithm(Algorithm):
    """
    Algorithm which selects points from a list of candidates.
    """

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self._candidates = self._get_candidates()
        self._num_candidates = len(self._candidates)
        self._i = 0
        self._exit = False

    def _get_candidates(self):
        return self.config.candidates()

    def _next(self):
        next = self._candidates[self._i]
        self._i += 1
        if self._i == self._num_candidates:
            self._exit = True
        return next

    @property
    def exit(self):
        return self._exit

class GridSearchConfig(AlgorithmConfig):
    points_per_dim = ConfigField(5)

# config_manager.register(GridSearchConfig)

@assign_config(GridSearchConfig)
class GridSearch(CandidateAlgorithm):
    def _get_candidates(self):
        return cartesian([np.linspace(self.domain.l[i], self.domain.u[i], self.config.points_per_dim) for i in range(self.domain.d)])



