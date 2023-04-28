from GPy.util.linalg import dtrtrs, tdot, dpotrs
from febo.utils import locate, get_logger

import math
import numpy as np

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression, GPHeteroscedasticRegression
from febo.utils.config import ConfigField, assign_config, config_manager
import GPy
from scipy.linalg import lapack
from scipy.optimize import minimize

logger = get_logger('model')

class GPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.1)
    calculate_gradients = ConfigField(True, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(False)
    optimize_var = ConfigField(True)
    bias = ConfigField(0)
    _section = 'models.gp'

# config_manager.register(GPConfig)

def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()



@assign_config(GPConfig)
class GP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """

    def __init__(self, domain):
        super(GP, self).__init__(domain)

        # the description of a kernel
        self.kernel = self._get_kernel()

        # calling of the kernel
        self.gp = self._get_gp()
        # number of data points
        self.t = 0
        self.kernel = self.kernel.copy()
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()
        self._Y = np.empty(shape=(0,1))
        self._beta_cached = 2
        self._bias = self.config.bias


    def _get_kernel(self):
        kernel = None
        for kernel_module, kernel_params in self.config.kernels:
            input_dim = self.domain.d
            if 'active_dims' in kernel_params:
                input_dim = len(kernel_params['active_dims'])
            kernel_part = locate(kernel_module)(input_dim=input_dim, **kernel_params)
            if kernel is None:
                kernel = kernel_part
            else:
                kernel += kernel_part
        return kernel

    def _beta(self):
        return self._beta_cached

    @property
    def scale(self):
        if self.gp.kern.name == 'sum':
            return sum([part.variance for part in self.gp.kern.parts])
        else:
            return np.sqrt(self.gp.kern.variance)

    @property
    def bias(self):
        return self._bias

    def _get_gp(self):
        return GPRegression(self.domain.d, self.kernel, noise_var=self.config.noise_var, calculate_gradients=self.config.calculate_gradients)

    def add_data(self, x, y):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        self._Y = np.vstack([self._Y, y])  # store unbiased data
        self.gp.append_XY(x, y - self._bias)

        self.t += y.shape[1]
        self._update_cache()


    def optimize(self):
        if self.config.optimize_bias:
            self._optimize_bias()
        if self.config.optimize_var:
            self._optimize_var()

        self._update_beta()


    def _update_cache(self):
        # if not self.config.calculate_gradients:
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

        self._update_beta()

    def _optimize_bias(self):
        self._bias = minimize(self._bias_loss, self._bias, method='L-BFGS-B')['x'].copy()
        self._set_bias(self._bias)
        logger.info(f"Updated bias to {self._bias}")

    def _bias_loss(self, c):
        # calculate mean and norm for new bias via a new woodbury_vector
        new_woodbury_vector,_= dpotrs(self._woodbury_chol, self._Y - c, lower=1)
        K = self.gp.kern.K(self.gp.X)
        mean = np.dot(K, new_woodbury_vector)
        norm = new_woodbury_vector.T.dot(mean)
        # loss is least_squares_error + norm
        return np.asscalar(np.sum(np.square(mean + c - self._Y)) + norm)

    def _set_bias(self, c):
        self._bias = c
        self.gp.set_Y(self._Y - c)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()

    def _update_beta(self):
        logdet = self._get_logdet()
        logdet_priornoise = self._get_logdet_prior_noise()
        self._beta_cached = (np.sqrt(2 * np.log(1 / self.delta) + (logdet - logdet_priornoise)) + self._norm()).item()
        # print(self._norm(), 'norm')
        # self._beta = 2
        # print(2 * mutual_information(self.gp), logdet - logdet_priornoise)

    def _optimize_var(self):
        # fix all parameters
        for p in self.gp.parameters:
            p.fix()

        if self.gp.kern.name == 'sum':
            for part in self.gp.kern.parts:
                part.variance.unfix()
        else:
            self.gp.kern.variance.unfix()
        self.gp.optimize()
        if self.gp.kern.name == 'sum':
            values = []
            for part in self.gp.kern.parts:
                values.append(np.asscalar(part.variance.values))
        else:
            values = np.asscalar(self.gp.kern.variance.values)

        logger.info(f"Updated prior variance to {values}")
        # unfix all parameters
        for p in self.gp.parameters:
            p.unfix()

    def _get_logdet(self):
        return 2.*np.sum(np.log(np.diag(self.gp.posterior._woodbury_chol)))

    def _get_logdet_prior_noise(self):
        return self.t * np.log(self.gp.likelihood.variance.values)


    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        x = np.atleast_2d(x)

        if self.config.calculate_gradients:
            mean,var = self.gp.predict_noiseless(x)
        else:
            mean,var = self._raw_predict(x)

        return mean + self._bias, var

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def predictive_var_scattered(self, X_cond, X_target):
        # X_cond = np.atleast_2d(X_cond)
        # X_target = np.atleast_2d(X_target)
        # KXX =
        # var_X, KXX = self._raw_predict_covar(X_cond, X_target)
        # if var_X_target is None:
        #     var_X_target = self.var(X_target)

        X_joined = np.vstack((X_target, X_cond))
        COV =  self.gp.predict_noiseless(X_joined, full_cov=True)[1]
        covar = COV[1:,0:1]
        var_X = np.diag(COV[1:,1:])
        return covar * covar, var_X

    def predictive_var(self, X_cond, X_target, S_X, var_X_target=None):
        # X_cond = np.atleast_2d(X_cond)
        # X_target = np.atleast_2d(X_target)
        # KXX =
        # var_X, KXX = self._raw_predict_covar(X_cond, X_target)
        # if var_X_target is None:
        #     var_X_target = self.var(X_target)

        var_X = self.var(X_cond)
        X_joined = np.vstack((X_target, X_cond))
        covar = self.gp.predict_noiseless(X_joined, full_cov=True)[1][1:,0:1]
        var_X_target = self.var(X_target)

        return var_X_target - covar * covar / (S_X*S_X + var_X)

        # TODO There are numerical problems with the version below
        # It fails te testcase with large lengthscale
        # print(var_Xcond)
        print("------")
        print(np.min(S_X*S_X))
        print(np.max(KXX*KXX/(S_X*S_X + var_X)))
        print(np.min(var_X))
        print('rho', np.min(S_X*S_X))
        print(np.min((S_X*S_X + var_X)))
        print(np.max(1/(S_X*S_X + var_X)))
        print((KXX*KXX/(S_X*S_X + var_X)).shape)
        print("------")
        return var_X_target - KXX * KXX / (S_X * S_X + var_X)

    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]
        self._update_cache()

    def sample(self, X=None):
        class GPSampler:
            def __init__(self, X, Y, kernel, var):
                self.X = X
                self.Y = Y
                self.N = var * np.ones(shape=Y.shape)
                self.kernel = kernel
                self.m = GPy.models.GPHeteroscedasticRegression(self.X, self.Y, self.kernel)
                self.m['.*het_Gauss.variance'] = self.N

            def __call__(self, X):
                X = np.atleast_2d(X)
                sample = np.empty(shape=(X.shape[0], 1))

                # iteratively generate sample values for all x in x_test
                for i, x in enumerate(X):
                    sample[i] = self.m.posterior_samples_f(x.reshape((1, -1)), size=1)

                    # add observation as without noise
                    self.X = np.vstack((self.X, x))
                    self.Y = np.vstack((self.Y, sample[i]))
                    self.N = np.vstack((self.N, 0))

                    # recalculate model
                    self.m = GPy.models.GPHeteroscedasticRegression(self.X, self.Y, self.kernel)
                    self.m['.*het_Gauss.variance'] = self.N  # Set the noise parameters to the error in Y

                return sample

        return GPSampler(self.gp.X.copy(), self.gp.Y.copy(), self.kernel, self.gp.likelihood.variance)

    def _raw_predict(self, Xnew):

        Kx = self.kernel.K(self._X, Xnew)
        mu = np.dot(Kx.T, self._woodbury_vector)

        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)

        Kxx = self.kernel.Kdiag(Xnew)
        tmp = lapack.dtrtrs(self._woodbury_chol, Kx, lower=1, trans=0, unitdiag=0)[0]
        var = (Kxx - np.square(tmp).sum(0))[:,None]
        return mu, var

    def _raw_predict_covar(self, Xnew, Xcond):
        Kx = self.kernel.K(self._X, np.vstack((Xnew,Xcond)))
        tmp = lapack.dtrtrs(self._woodbury_chol, Kx, lower=1, trans=0, unitdiag=0)[0]

        n = Xnew.shape[0]
        tmp1 = tmp[:,:n]
        tmp2 = tmp[:,n:]

        Kxx = self.kernel.K(Xnew, Xcond)
        var = Kxx - (tmp1.T).dot(tmp2)

        Kxx_new = self.kernel.Kdiag(Xnew)
        var_Xnew = (Kxx_new - np.square(tmp1).sum(0))[:,None]
        # var = var
        return var_Xnew, var

    def _norm(self):
        norm = self._woodbury_vector.T.dot(self.gp.kern.K(self.gp.X)).dot(self._woodbury_vector)
        return (np.sqrt(norm)).item()



    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['gp'] # remove the gp from state dict to allow pickling. calculations are done via the cache woodbury/cholesky
        return self_dict


class HeteroscedasticGP(GP):

    def _get_gp(self):
         return GPHeteroscedasticRegression(self.domain.d, self.kernel, calculate_gradients=self.config.calculate_gradients)

    def add_data(self, x, y, s):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        s = np.atleast_2d(s)
        self._Y = np.vstack([self._Y, y])
        self.gp.append_XY(x,y - self._bias, s)
        self.t += y.shape[1]
        self._update_cache()


    # def set_data(self, X, Y, S, append=True):
    #     if append:
    #         X = np.concatenate((self.gp.X, X))
    #         Y = np.concatenate((self.gp.Y, Y))
    #         S = np.concatenate((self.gp.S, S))
    #     self.gp.set_XY(X, Y - self._bias, S)
    #     self.t = X.shape[0]
    #     self._update_cache()

    @property
    def requires_std(self):
        return True

    def _get_logdet_prior_noise(self):
        return np.sum(2*np.log(self.gp.S))

# def mutual_information(gp):
#     """Return the mutual information obtained by a GP.
#
#     Parameters
#     ----------
#     gp : GPy.models.GP
#
#     Returns
#     -------
#     mutual_information : float
#     """
#     noise_var = gp.likelihood.variance.values
#     cholesky = gp.posterior.woodbury_chol
#
#     # 0.5 * log(|I + K / noise_var|)
#     # L = np.linalg.cholesky(I * noise_var + K)
#     # L has eigenvalues on diagonal, but need to rescale by sqrt(noise)
#     # L L^T = K + noise_var I  --> L' L'^T = K / noise_var + I
#     # with L' = L / noise_std
#     # Thus 0.5 * log(|I + K / noise|) = 0.5 * log(prod(diag(L')) ** 2)
#     # = log(prod(diag(L'))) = sum(log(diag(L')))
#     mutual_information = np.log(np.diag(cholesky) / np.sqrt(noise_var))
#
#     return np.sum(mutual_information)
