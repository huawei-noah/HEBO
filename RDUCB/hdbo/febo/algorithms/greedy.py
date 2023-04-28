from febo.algorithms.safety import SafetyMixin
from febo.algorithms.model import ModelMixin
from febo.algorithms.acquisition import AcquisitionAlgorithm


class Greedy(ModelMixin, AcquisitionAlgorithm):
    """
    Implements the Upper Confidence Bound (UCB) algorithm.
    """

    def initialize(self, **kwargs):
        super(Greedy, self).initialize(**kwargs)

    def acquisition(self, X):
        X = X.reshape(-1, self.domain.d)
        return -(self.model.mean(X)-self.model.bias)/self.model.scale

class SafeGreedy(SafetyMixin, Greedy):
    pass