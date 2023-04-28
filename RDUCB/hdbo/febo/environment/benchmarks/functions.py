from febo.utils.config import ConfigField, assign_config
from .benchmarks import BenchmarkEnvironment, BenchmarkEnvironmentConfig
import numpy as np
from febo.environment.domain import DiscreteDomain, ContinuousDomain



class FiniteLinearBandit(BenchmarkEnvironment):
    """
    Quick sketch of a finite linear bandit
    """

    def __init__(self, path=None):
        super().__init__(path=path)
        self._theta = np.ones(self.config.dimension)
        self._theta = self._theta / np.linalg.norm(self._theta)

        np.random.seed(self.seed)
        self._domain_points = np.random.multivariate_normal(np.zeros(self.config.dimension),
                                                            np.eye(self.config.dimension),
                                                            size=self.config.num_domain_points)

        np.random.seed()  # reset to random state

        self._domain_points = self._domain_points / np.maximum(np.linalg.norm(self._domain_points, axis=-1),
                                                               np.ones(self.config.num_domain_points)).reshape(-1, 1)
        self._domain = DiscreteDomain(self._domain_points)
        self._max_value = self._get_max_value()

    @property
    def name(self):
        return "Finite Linear Bandit"

    @property
    def _requires_random_seed(self):
        return True

    def f(self, x):
        return np.dot(x, self._theta)

    def _get_max_value(self):
        return np.max(self.f(self._domain_points))


class Camelback1D(BenchmarkEnvironment):
    """
    1d Test Function
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._x = np.array([1.])
        self._max_value = 1.0026469
        self._domain = ContinuousDomain(np.array([-1]), np.array([2]))

    def f(self, x):
        return np.exp(-np.square(x-1.5)/0.05) + 1.98/(1+np.square(x-0.5)) - 1


class Camelback(BenchmarkEnvironment):
    """
    Camelback benchmark function.
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._x0 = np.array([0.5, 0.2])
        self._x0 = np.array([-0.12977758051079197, 0.2632096107305229])
        self._max_value = 1.03162842
        self._domain = ContinuousDomain(np.array([-2,-1]), np.array([2,1]))

    def f(self, x):
        x = np.atleast_2d(x)
        xx = x[:,0]
        yy = x[:,1]
        y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
        return np.maximum(-y, -2.5)


class Camelmod(BenchmarkEnvironment):
    """
    Camelmod benchmark function with local minima
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._x0 = np.array([0., 0.])
        self._max_value = 1.
        self._domain = ContinuousDomain(np.array([-0.5,-0.5]), np.array([0.5,0.5]))

    def f(self, X):
        X = np.atleast_2d(X)
        X, Y = X[:, 0], X[:, 1]
        return ((8 * X) ** 4 - 16. * (8 * X) ** 2 + 5 * (8 * X) + (8 * Y) ** 4 - 16. * (8 * Y) ** 2 + 5 * (
                    8 * Y)) / -156.66466273


class GaussianConfig(BenchmarkEnvironmentConfig):
    initial_value = ConfigField(0.1)
    _section = 'environment.benchmark.gaussian'

@assign_config(GaussianConfig)
class Gaussian(BenchmarkEnvironment):
    """
    Camelback benchmark function.
    """
    def __init__(self, path=None):
        super().__init__(path)
        ones = np.ones(self.config.dimension)

        self._dist_initial = np.sqrt(np.log(1 / self.config.initial_value) / 4)
        self._x0 = self._dist_initial * ones / np.sqrt(self.config.dimension)
        self._max_value = 1.0
        self._domain = ContinuousDomain(-ones, ones)

    def _get_random_initial_point(self):
        dir = np.random.normal(size=self.config.dimension)
        return self._dist_initial * dir / np.linalg.norm(dir)

    def f(self, X):
        X = np.atleast_2d(X)
        Y = np.exp(-4*np.sum(np.square(X), axis=1))
        return Y


class Quadratic(BenchmarkEnvironment):
    """
    Camelback benchmark function.
    """
    def __init__(self, path=None):
        super().__init__(path)
        ones = np.ones(self.config.dimension)
        self._x0 = 0.5*ones/np.sqrt(self.config.dimension)
        self._max_value = 1.0
        self._domain = ContinuousDomain(-ones, ones)

    def f(self, X):
        X = np.atleast_2d(X)
        Y = 2*np.sum(np.square(X), axis=1)
        return 1 - Y

class CamelbackEmbedded(BenchmarkEnvironment):
    """
    Camelback benchmark function.
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._max_value = 1.03162842
        d = self.config.dimension
        if d <= 2:
            raise Exception("Need dimension at least 3 to create embedded version of Camelback")
        self._x0 = np.array([0.5, 0.2] + [0.]*(d-2))
        self._domain = ContinuousDomain(np.array([-2,-1] + [-1]*(d-2)), np.array([2,1]+ [1]*(d-2)))

    def f(self, x):
        xx = x[0]
        yy = x[1]
        y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
        return -y


class LinSin1D(BenchmarkEnvironment):
    """
    d=1 benchmark function
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._x = np.array([15])
        self._max_value = 1.25375424 # determined using scipy.minimze
        self._domain = ContinuousDomain(np.array([-20]), np.array([20]))

    def f(self, X):
        return 10. + 0.05*X + np.sin(X-5)/(X-5) - 10


class CosUnique1D(BenchmarkEnvironment):
    """
    d=1 benchmark function
    """
    def __init__(self, path=None):
        super().__init__(path)
        self._x = np.array([0.1])
        self._max_value = 1.1 # at 0.5
        self._domain = ContinuousDomain(np.array([0.]), np.array([1]))

    def f(self, X):
        return  -np.cos(10*np.pi*X) + 0.1 - 0.1*np.abs(X-0.5)
#
# class DSafetyConstraintsEnv(BenchmarkEnvironment):
#     """ implements a _get_saftey_constraint methods, which returns self.domain_dimension safety constraints. """
#     def _get_safety_constraints(self, x):
#         constraints = []
#         # if self.config.TEST_ENVS_USE_SAFETY_CONSTRAINTS:
#         #     for i in range(self.domain_dimension):
#         #         constraints.append(0.5 * math.cos(10 * x[i] - i % 2) - 1 + 2 * x[max(0,i-1)])
#
#         return np.array(constraints)
#
#
#     def initialize(self):
#         if self.config.TEST_ENVS_USE_SAFETY_CONSTRAINTS:
#             # self._num_safety_constraints = self.domain_dimension
#             self._lower_bound_objective_value = 0.3
#         else:
#             self._num_safety_constraints = 0
#             self._lower_bound_objective_value = None
#         super(DSafetyConstraintsEnv, self).initialize()
#
# class Simple2D(DSafetyConstraintsEnv):
#
#     @property
#     def domain_dimension(self):
#         return 2
#
#     def _f(self, x):
#         return math.sin(.3*x[0]) + .3*math.cos(1.3*x[1]*x[0])


# class Simple1D(DSafetyConstraintsEnv):
#
#     @property
#     def domain_dimension(self):
#         return 1
#
#     def f(self, x):
#         return math.sin(5*x[0]) + 0.5 *math.cos(10*x[0])




# class Michalewicz(DSafetyConstraintsEnv):
#
#     def __init__(self, *args, **kwargs):
#         super(Michalewicz, self).__init__(*args, **kwargs)
#         self.d = 2
#
#     def initialize(self):
#         super(Michalewicz, self).initialize()
#
#         # overwrite initial point
#         self._current_x = np.array([0.2]*self.d)
#         self.set_parameters(self._current_x) # refresh objective value
#         self._lower_bound_objective_value =  -2.0 if self.config.TEST_ENVS_USE_SAFETY_CONSTRAINTS else None
#
#     @property
#     def domain_dimension(self):
#         return self.d
#
#     def f(self, x):
#         x = x / np.pi
#         (d) = x.shape
#         ar = np.arange(1,self.d+1,1)
#         sum_ = np.sin(x) * np.power((np.sin(ar * np.power(x, 2) / np.pi)), (2*d))
#         sum_ = np.sum(sum_)
#         return -0.5*sum_
#
# class SineMultiDim(DSafetyConstraintsEnv):
#
#     def __init__(self, *args, **kwargs):
#         super(SineMultiDim, self).__init__(*args, **kwargs)
#         self._dim = 2
#
#     def initialize(self):
#         super(Camelback, self).initialize()
#
#         # overwrite initial point
#         self._current_x = np.array([0.2] * self.domain_dimension)
#         self.set_parameters(self._current_x)  # refresh objective value
#         self._lower_bound_objective_value = None
#
#     def set_dimension(self, dim):
#         self._dim = dim
#
#     @property
#     def domain_dimension(self):
#         return 2
#
#     def f(self, x):
#         y = -np.sin(np.sum(x**2)) + 1
#         return y
#
