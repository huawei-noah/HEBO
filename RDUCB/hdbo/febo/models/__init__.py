"""
The `model` package implements a base class for custom models and provides common defaults like a linear model and Gaussian processes.

Models
------

.. autosummary::
   :template: template.rst
   :toctree:

   Model
   ConfidenceBoundModel
   LinearLeastSquares
   GP
   GPConfig
   StandaloneGP
   FourierFeatures
"""

from .model import Model, ConfidenceBoundModel

from .lls import LinearLeastSquares, WeightedLinearLeastSquares
from .gp import GP, GPConfig