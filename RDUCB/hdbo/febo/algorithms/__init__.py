"""
The `algorithm` package implements a base class for algorithms. Common algorithms are included as well as
algorithms which are defined through an acquisition function. Algorithms can be configured to use different models,
like linear estimators or GPs.

Algorithms
----------
.. autosummary::
   :template: template_private.rst
   :toctree:

   Algorithm

   :template: template.rst
   :toctree:

   AcquisitionAlgorithm
   ModelMixin
   UCB
   Random

"""


from .algorithm import Algorithm, AlgorithmConfig
from .model import ModelMixin
from .acquisition import AcquisitionAlgorithm
from .ucb import UCB, SafeUCB
from .random import Random
from .candidate import CandidateAlgorithm
from .ts import ThompsonSampling
from .greedy import Greedy, SafeGreedy
from .neldermead import NelderMead
from .subdomainbo import AscentLineBO, SafeAscentLineBO, CoordinateLineBO, SafeCoordinateLineBO, RandomLineBO, SafeRandomLineBO