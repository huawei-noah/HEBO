from .acquisition import AcquisitionAlgorithm
from .model import ModelMixin


class ThompsonSampling(ModelMixin, AcquisitionAlgorithm):
    """
	Implements the ThompsonSampling algorithm.
	"""

    def acquisition_init(self):
        self._sample = self.model.sample()

    def acquisition(self, X):
        return self._sample(X)

    def acquisition_grad(self, x):
        raise NotImplementedError

# class WeightedThompsonSampling(ModelWithNoiseMixin, AcquisitionAlgorithm):
#     """
#     Implements the ThompsonSampling algorithm.
#     """
#
#     def acquisition_init(self):
#         self._sample = self.model.sample()
#
#     def acquisition(self, X):
#         return -self._sample(X)
#
#     def acquisition_grad(self, x):
#         raise NotImplementedError
