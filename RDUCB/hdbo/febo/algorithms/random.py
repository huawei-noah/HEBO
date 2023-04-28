from .algorithm import Algorithm
import numpy as np



class Random(Algorithm):
    """
    Algorithm which selects random points in the domain.
    """

    def _next(self):
        if self.domain.is_continuous:
            next = np.random.uniform(size=self.domain.d)*self.domain.range + self.domain.l
        else:
            next = self.domain.points[np.random.choice(self.domain.num_points)]
        return next
