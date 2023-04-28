from febo.models import Model, ConfidenceBoundModel
import numpy as np



class LinearModel(ConfidenceBoundModel):


    def __init__(self, domain):
        super().__init__(domain)
        self._V = np.eye(self.domain.d)
        self._Y = np.zeros(self.domain.d)
        self._theta = np.zeros(self.domain.d)
        self._n = 0
        self._update_cached()

    def add_data(self, X, Y):
        """ just updating cache here and counting, adding datapoint needs to be done in child class"""
        self._update_cached()
        self._n += 1

    def _update_cached(self):
        self._theta = np.linalg.solve(self._V, self._Y)
        self._detV = np.linalg.det(self.V)
        self._beta_t = np.sqrt(np.log(self.detV) - 2 * np.log(self.delta)) + 1

    @property
    def V(self):
        return self._V

    def mean(self, X):
        X = np.atleast_2d(X)
        return X.dot(self._theta).reshape(-1,1)

    def var(self, X):
        X = np.atleast_2d(X)
        return np.sum(X * (np.linalg.solve(self.V, X.T).T), axis=1).reshape(-1,1)

    def mean_var(self, X):
        X = X.reshape(-1, self.domain.d)
        return self.mean(X), self.var(X)

    #TODO: Update Iterativly
    @property
    def detV(self):
        return self._detV

    # @property
    # def beta(self):
    #     return self._beta_t

    #TODO: Caching/Iterative Update
    @property
    def V_inv(self):
        return np.linalg.inv(self.V)

    def _beta(self):
        return self._beta_t

    def sample(self, X=None):
        sampled_theta = np.random.multivariate_normal(self._theta, np.linalg.inv(self._V))
        def sample_f(X):
            X = np.atleast_2d(X)
            return X.dot(sampled_theta).reshape(-1, 1)
        return sample_f

class LinearLeastSquares(LinearModel):

    def add_data(self, X, Y):
        X = X.reshape((-1, self.domain.d))
        Y = Y.reshape((-1, 1))

        self._Y += X.T.dot(Y).flatten()
        for x,y in zip(X,Y):
            x = x.reshape(1,-1)
            self._V += x.T.dot(x)

        super(LinearLeastSquares, self).add_data(X,Y)



class WeightedLinearLeastSquares(LinearModel):

    @property
    def requires_std(self):
        return True

    def add_data(self, X, Y, rho):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        rho = np.atleast_2d(rho)

        r2 = rho * rho
        self._Y += X.T.dot(Y/r2).flatten()
        for x, y, r in zip(X, Y, r2):
            x = x.reshape(1, -1)
            self._V += x.T.dot(x)/r

        super(WeightedLinearLeastSquares, self).add_data(X, Y)

    def predictive_var(self,X_cond ,X,  rho_cond, var_x=None):
        X = np.atleast_2d(X)
        rho_cond = np.atleast_2d(rho_cond)
        X_cond = np.atleast_2d(X_cond)

        X_V_X_cond = np.sum(X_cond * (np.linalg.solve(self._V, X.T).T), axis=1).reshape(-1,1)
        return self.var(X) - X_V_X_cond * X_V_X_cond /(rho_cond * rho_cond + self.var(X_cond))
