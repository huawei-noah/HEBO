import numpy as np
from resources.casmopolitan.test_funcs.base import TestFunction

# Func2C and Func3C as appeared in CoCaBO


class Func2C(TestFunction):
    """Func2C is a mixed categorical and continuous function. The first 2 dimensions are categorical,
    with possible 3 and 5 possible values respectively. The last 2 dimensions are continuous"""

    """
    Global minimum of this function is at
    x* = [1, 1, -0.0898/2, 0.7126/2]
    with f(x*) = -0.2063
    """
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, normalize=False):
        # Specifies the indices of the dimensions that are categorical and continuous, respectively
        super(Func2C, self).__init__(normalize)
        self.categorical_dims = np.array([0, 1])
        self.continuous_dims = np.array([2, 3])
        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.n_vertices = np.array([3, 5])
        self.config = self.n_vertices
        # Specfies the range for the continuous variables
        self.lb = np.array([-1, -1])
        self.ub = np.array([1, 1])
        self.lamda = lamda

        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, X, normalize=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        res = np.zeros((N, ))
        X_cat = X[:, self.categorical_dims]
        X_cont = X[:, self.continuous_dims]
        X_cont = X_cont * 2

        for i, X in enumerate(X):
            if X_cat[i, 0] == 0:
                res[i] = myrosenbrock(X_cont[i, :])
            elif X_cat[i, 0] == 1:
                res[i] = mysixhumpcamp(X_cont[i, :])
            else:
                res[i] = mybeale(X_cont[i, :])

            if X_cat[i, 1] == 0:
                res[i] += myrosenbrock(X_cont[i, :])
            elif X_cat[i, 1] == 1:
                res[i] += mysixhumpcamp(X_cont[i, :])
            else:
                res[i] += mybeale(X_cont[i, :])
        res += self.lamda * np.random.rand(*res.shape)
        return res

    def sample_normalize(self, size=None):
        from bo.localbo_utils import latin_hypercube, from_unit_cube
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x_cat = np.array([np.random.choice(self.config[_]) for _ in range(self.categorical_dims.shape[0])])
            x_cont = latin_hypercube(1, self.continuous_dims.shape[0])
            x_cont = from_unit_cube(x_cont, self.lb, self.ub).flatten()
            x = np.hstack((x_cat, x_cont))
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)


class Func3C(TestFunction):
    """
    Func3C is a simlar function, but with 3 categorical variables (each of which is binary) and 2 continuous variables
    """
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, normalize=False):
        super(Func3C, self).__init__(normalize)
        self.categorical_dims = np.array([0, 1, 2])
        self.continuous_dims = np.array([3, 4])
        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.n_vertices = np.array([2, 2, 2])
        self.config = self.n_vertices
        # Specfies the range for the continuous variables
        self.lb = np.array([-1, -1])
        self.ub = np.array([1, 1])
        self.lamda = lamda
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, X, normalize=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        res = np.zeros((N, ))
        X_cat = X[:, self.categorical_dims]
        X_cont = X[:, self.continuous_dims]
        X_cont = X_cont * 2

        for i, X in enumerate(X):
            if X_cat[i, 0] == 0:
                res[i] = myrosenbrock(X_cont[i, :])
            elif X_cat[i, 0] == 1:
                res[i] = mysixhumpcamp(X_cont[i, :])
            elif X_cat[i, 0] == 2:  # should never be activated
                res[i] = mybeale(X_cont[i, :])

            if X_cat[i, 1] == 0:
                res[i] += myrosenbrock(X_cont[i, :])
            elif X_cat[i, 1] == 1:
                res[i] += mysixhumpcamp(X_cont[i, :])
            else: # should never be activated
                res[i] += mybeale(X_cont[i, :])

            if X_cat[i, 2] == 0:
                res[i] += 5 * mysixhumpcamp(X_cont[i, :])
            elif X_cat[i, 2] == 1:
                res[i] += 2 * myrosenbrock(X_cont[i, :])
            else: # should never be activated
                res[i] += X_cont[i, 2] * mybeale(X_cont[i, :])

        res += self.lamda * np.random.rand(*res.shape)
        return res

    def sample_normalize(self, size=None):
        from bo.localbo_utils import latin_hypercube, from_unit_cube
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x_cat = np.array([np.random.choice(self.config[_]) for _ in range(self.categorical_dims.shape[0])])
            x_cont = latin_hypercube(1, self.continuous_dims.shape[0])
            x_cont = from_unit_cube(x_cont, self.lb, self.ub).flatten()
            x = np.hstack((x_cat, x_cont))
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)


class Ackley53(TestFunction):
    problem_type = 'mixed'

    # Taken and adapted from the the MVRSM codebase
    def __init__(self, lamda=1e-6, normalize=False):
        super(Ackley53, self).__init__(normalize)
        self.categorical_dims = np.arange(0, 50)
        self.continuous_dims = np.array([50, 51, 52])
        self.dim = len(self.continuous_dims) + len(self.categorical_dims)
        self.n_vertices = 2 * np.ones(len(self.categorical_dims), dtype=int)
        self.config = self.n_vertices
        self.lamda = lamda
        # specifies the range for the continuous variables
        self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])

    @staticmethod
    def _ackley(X):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(X), axis=1) / 53))
        cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(X)) / 53, axis=1))
        result = a + np.exp(1) + sum_sq_term + cos_term
        return result

    def compute(self, X, normalize=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        X[:, self.categorical_dims] = np.round(X[:, self.categorical_dims])
        result = self._ackley(X)
        return result + self.lamda * np.random.rand(*result.shape)


# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10


# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50


if __name__ == '__main__':
    f = Func3C()
    print(f.sample_normalize(10))
