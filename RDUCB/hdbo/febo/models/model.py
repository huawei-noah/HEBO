import numpy as np
from febo.utils import locate, get_logger
from febo.utils.config import ConfigField, Config, assign_config, Configurable

logger = get_logger('model')

class Model(Configurable):
    """
    A base class to implement custom models.
    """

    def __init__(self, domain):
        """


        Args:
            d (int): input dimension
        """
        self.domain = domain

    def mean(self, X):
        """
        Calculate predicted mean at X.

        Args:
            X: input narray of shape (N,d)

        Returns:
            Predicted mean, narray of shape (N,1)

        """
        raise NotImplementedError

    def var(self, X):
        """
        Calculate predicted variance at X

        Args:
            X: input narray of shape (N,d)

        Returns:
            Predicted variance, narray of shape (N,1)

        """
        raise NotImplementedError

    def std(self, x):
        return np.sqrt(self.var(x))

    def mean_var(self, x):
        """

        Calculate predicted mean and variance at X

        Args:
            X: input narray of shape (N,d)

        Returns:
            (mean, var) Predicted mean and variance, each narray of shape (N,1)

        """
        raise NotImplementedError

    def mean_var_grad(self, x):
        """
        Args:
            x:

        Returns: TODO: (mean,var,mean_grad, var_grad) or just gradients?

        """
        raise NotImplementedError

    def add_data(self, X, Y):
        """
        Add data to the model.

        Args:
            X: input narray of shape (N,d)
            Y: observation narray of shape (N,1)

        """
        raise NotImplementedError

    def sample(self, X=None):
        """
        Returns a sample form the posterior. It should return a function ``def my_sample(X):``

        Args:
            X: if specified, the sample function is only valid for points X

        Returns (function):

        """
        raise NotImplementedError


    def info(self):
        """
        Return Information about the Gaussian Process

        :return: string
        """


    @property
    def requires_std(self):
        return False

    def predictive_var(self, X, X_cond):
        """
        Returns the var(X|X_cond)

        Args:
            X:
            X_cond:

        Returns:

        """

        raise NotImplementedError

class ModelConfig(Config):
    delta = ConfigField(0.05)
    beta = ConfigField(default=2, allow_none=True)
    _section = "model"

@assign_config(ModelConfig)
class ConfidenceBoundModel(Model):
    """
    'ConfidenceBoundModel' extends the 'Model' by using the variance to calculate confidence bounds
    of the form [mean(x) - beta*std(x), mean(x) + beta*std(x)].
    """

    def __init__(self, domain):
        super().__init__(domain)

        self._beta_function = None
        if not self.config.beta is None:
            if isinstance(self.config.beta, str):
                self._beta_function = locate(self.config.beta)
                self.__beta = self.__beta_function
                logger.info(f"Using beta function={self.config.beta} .")
            elif isinstance(self.config.beta, (int, float)):
                def _beta():
                    return self.config.beta
                self.__beta = _beta
                logger.info(f"Using beta={self.config.beta} .")

    @property
    def delta(self):
        """
        Confidence bounds a calculated at level 1-delta.

        """
        return self.config.delta

    @property
    def beta(self):
        """
        Scaling Factor beta
        """
        return self.__beta()

    def __beta(self):
        return self._beta()

    def __beta_function(self):
        return self._beta_function(self)

    def _beta(self):
        """
        Model default value for beta
        """
        raise NotImplementedError

    def ucb(self, x):
        """
        Upper Confidence Bound

        Args:
            x:

        Returns:

        """
        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        return mean + self.beta*std

    def lcb(self, x):
        """
        Lower Confidence Bound

        Args:
            x:

        Returns:

        """
        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        return mean - self.beta*std

    def ci(self, x):
        """
        Confidence Interval

        Args:
            x:

        Returns:

        """
        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        beta = self.beta
        return mean - beta * std, mean + beta * std

    #TODO: Add gradient methods

#
# class ScaleShiftModel(Model):
#     """ takes a model M, returns a*M + b"""
#
#     def __init__(self, model, a, b):
#         if not (a == -1):
#             # Be aware that this is made general, the variance needs to be scaled
#             # Further, only if a < 0, ucb and lcb swap
#             raise Exception("Can only scale by -1 for now.")
#
#         self.a = a
#         self.b = b
#         self.m = model
#
#         super(ScaleShiftModel, self).__init__(self.m.config, self.m.domain_dimension)
#
#     def cb(self, x):
#         lcb, ucb = self.m.cb(x)
#         # swap lcb and ucb because a = -1
#         return ucb * self.a + self.b, lcb * self.a + self.b
#
#     def mean_var(self, x):
#         mean, var = self.m.mean_var(x)
#         return self.a * mean + self.b, var
#
#     def ucb(self, x):
#         # if we flip the function, the previous lcb is now ucb.
#         return self.m.lcb(x) * self.a + self.b
#
#     def lcb(self, x):
#         return self.m.ucb(x) * self.a + self.b
#
#     @property
#     def beta(self):
#         return self.m.beta
#
#     def var(self, x):
#         return self.m.var()
#
#     def mean(self, x):
#         return self.m.mean() * self.a + self.b
#
#     def std(self, x):
#         return super().std(x)
#
#     def mean_var_grad(self, x):
#         dmu,dvar= self.m.mean_var_grad(x)
#         return self.a * dmu, dvar
#
#     def add_data(self, x, y):
#         raise Exception("You should add data directly to original model!")
#
#     def set_data(self, X, Y, append=True):
#         raise Exception("You should add data directly to original model!")



