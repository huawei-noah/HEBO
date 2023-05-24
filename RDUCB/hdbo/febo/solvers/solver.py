from febo.utils.config import Configurable

class Solver(Configurable):

    def __init__(self, domain, initial_x=None):
        self._domain = domain
        self.initial_x = initial_x

    def minimize(self, f):
        """
            optimize f over domain
            if self.requires_gradients = True, fun should return a tuple of (y,grad)

         """
        raise NotImplementedError

    @property
    def requires_gradients(self):
        raise NotImplementedError

    @property
    def requires_safety(self):
        raise NotImplementedError


