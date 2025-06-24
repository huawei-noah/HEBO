from abc import abstractmethod
from typing import Optional

import numpy as np
import torch


class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'

    def __init__(self, normalise: bool = True, **kwargs) -> None:
        self.normalise = normalise
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self) -> None:
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, 'int_constrained_dims must be a subset of the continuous_dims, ' \
                                                 'but continuous_dims is not supplied!'
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), "all continuous dimensions with integer " \
                                                           "constraint must be themselves contained in the " \
                                                           "continuous_dimensions!"

    @abstractmethod
    def compute(self, x: np.ndarray, normalise: Optional[bool] = None) -> torch.tensor:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> torch.tensor:
        return self.compute(*args, **kwargs)
