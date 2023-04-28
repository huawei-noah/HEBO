"""

Plotting
--------

.. autosummary::
   :template: template.rst
   :toctree:

   plot.CumulativeRegretPlot

"""

from . import utilities
from .regret import Regret, SimpleRegret, InferenceRegret
from .time import Time, CumulativeTime
from .plot import Plot, DataPlot