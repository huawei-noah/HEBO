"""

Optimizers
----------

.. autosummary::
   :template: template.rst
   :toctree:

   Solver
   ScipySolver
   CandidateSolver
   GridSolver

"""

from .solver import Solver
from .scipy import ScipySolver
from .candidate import CandidateSolver, GridSolver, FiniteDomainSolver
