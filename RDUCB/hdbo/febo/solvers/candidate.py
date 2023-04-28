import numpy as np

from .solver import Solver
from febo.utils import cartesian
from febo.utils.config import ConfigField, assign_config, Config, config_manager


class CandidateSolver(Solver):
    def __init__(self, domain, candidates):
        """
        Deviates from parent argument interface.
        Args:
            domain:
            candidates:
        """
        super(CandidateSolver, self).__init__(domain)
        self.candidates = candidates

    def minimize(self, f):
        res = f(self.candidates)
        index = np.argmin(res)
        best_x = self.candidates[index]
        return best_x, res[index]

    @property
    def requires_gradients(self):
        return False

    @property
    def requires_safety(self):
        return False

class FiniteDomainSolver(CandidateSolver):
    def __init__(self, domain, initial_x=None):
        super().__init__(domain, domain.points)

class GridSolverConfig(Config):
    points_per_dimension = ConfigField(300)
    _section = 'solver.grid'

@assign_config(GridSolverConfig)
class GridSolver(CandidateSolver):

    def __init__(self, domain, initial_x=None, points_per_dimension=None):
        if points_per_dimension is None:
            points_per_dimension = self.config.points_per_dimension

        arrays = [np.linspace(l, u, points_per_dimension).reshape(points_per_dimension, 1) for (l, u) in
                  zip(domain.l, domain.u)]
        grid = cartesian(arrays)
        super(GridSolver, self).__init__(domain, candidates=grid)
