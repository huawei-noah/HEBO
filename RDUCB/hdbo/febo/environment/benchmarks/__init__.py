"""

Benchmarks
----------

.. autosummary::
   :template: template.rst
   :toctree:

   BenchmarkEnvironment
   Camelback

"""

from .functions import FiniteLinearBandit, Camelback
from .noise import RandomNoise, NoiseFunction
from .benchmarks import BenchmarkEnvironment, BenchmarkEnvironmentConfig