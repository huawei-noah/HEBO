import GPy
import numpy as np
from GPy.inference.latent_function_inference.posterior import PosteriorExact as Posterior
from GPy.util.linalg import pdinv, dpotrs, tdot, dtrtrs, dpotri, symmetrify
from GPy.util import diag
from scipy import stats

log_2_pi = np.log(2*np.pi)

def __setattr_patch__(self, name, val):
    # override the default behaviour, if setting a param, so broadcasting can by used
    # fix: ignore names stasrting with '_', this prevents parameter_names(...) to be called, when
    # private members of a Parameterized class are called (during some tests, this made for 3% runtime)
    if name[0] != '_' and hasattr(self, "parameters"):
        pnames = self.parameter_names(False, adjust_for_printing=True, recursive=False)
        if name in pnames:
            param = self.parameters[pnames.index(name)]
            param[:] = val
            return
    return super(Parameterized, self).__setattr__(name, val)

from paramz import Parameterized
Parameterized.__setattr__ = __setattr_patch__

class IncrementalKernelCacheMixin:
    """
    Adds self._K, a cache of the kernel matrix.
    Even though GPy already caches the kernel matrix, on incremental updates, the kernel is re-evaluated at all entries,
    which turns out to be a major slow-down.
    This mixin adds incremental updates for kernel caching.
    """
    def __init__(self, *args, **kwargs):
        self._K = None
        super(IncrementalKernelCacheMixin, self).__init__(*args, **kwargs)

    def update_incremental(self, X):
        K_tmp = self.K(X, X[-1:])
        K_inc = K_tmp[:-1]
        K_inc2 = K_tmp[-1:]
        self._K = np.block([[self._K, K_inc], [K_inc.T, K_inc2]])

    def update_non_incremental(self, X):
        self._K = self.K(X)

class IncrementalKernelGradientsMixin:
    """
    Mixin to incrementally update kernel gradients for isotropic kernels
    """
    def __init__(self, *args, **kwargs):
        super(IncrementalKernelGradientsMixin, self).__init__(*args, **kwargs)

    def update_gradients_incremental(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self._K * dL_dK) / self.variance

        dK_dr_via_X_tmp = self.dK_dr_via_X(X, X[-1:])
        dK_dr_via_X_inc = dK_dr_via_X_tmp[:-1]
        dK_dr_via_X_inc2 = dK_dr_via_X_tmp[-1:]
        self._dK_dr_via_X = np.block([[self._dK_dr_via_X.copy(), dK_dr_via_X_inc], [dK_dr_via_X_inc.T, dK_dr_via_X_inc2]])

        # now the lengthscale gradient(s)
        dL_dr = self._dK_dr_via_X * dL_dK
        if self.ARD:

            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            if GPy.util.config.config.getboolean('cython', 'working'):
                self.lengthscale.gradient = self._lengthscale_grads_cython(tmp, X, X2)
            else:
                self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
        else:
            r = self._scaled_dist(X, X2)
            self.lengthscale.gradient = -np.sum(dL_dr * r) / self.lengthscale

    def update_gradients_full(self, dL_dK, X, X2=None):
        super().update_gradients_full(dL_dK,X,X2)
        self._dK_dr_via_X = self.dK_dr_via_X(X, X2)

# we add our mixins directly onto GPy's kernel definitions.
GPy.kern.RBF.__bases__ = (IncrementalKernelCacheMixin, IncrementalKernelGradientsMixin) +  GPy.kern.RBF.__bases__
GPy.kern.Matern32.__bases__ = (IncrementalKernelCacheMixin, IncrementalKernelGradientsMixin) +  GPy.kern.Matern32.__bases__
GPy.kern.Matern52.__bases__ = (IncrementalKernelCacheMixin, IncrementalKernelGradientsMixin) +  GPy.kern.Matern52.__bases__
GPy.kern.Add.__bases__ = (IncrementalKernelCacheMixin, ) +  GPy.kern.Add.__bases__

class GP(GPy.core.GP):
    """
        This class extends GPy to incremental updates by adding an appendXY method.
        We also add a switch to turn of calculatation of gradients.
        It also contains a bit of ugly hacking to allow the GPRegression class to be instantiated without data,
        by adding a fake data point to allow initialitation, which is removed on the consquetive call of either
        set_XY or append_XY. """

    def __init__(self, input_dim, kernel, likelihood, mean_function=None, name='gp', Y_metadata=None, normalizer=False, calculate_gradients=True):

        X = np.zeros(input_dim).reshape(1,-1) # add some data to make GPy happy
        Y = np.zeros(1).reshape(1,-1) # add some data to make GPy happy

        inference_method = ExactGaussianInferenceIncremental() # add our own incremental inference method
        self._update_incremental = False
        self._calculate_gradients = calculate_gradients
        self._input_dim = input_dim
        # inference_method = None
        # inference_method = None
        # extend_instance(kernel, IncrementalKernelMixin)
        super().__init__(X,Y, kernel, likelihood, mean_function, inference_method, name, Y_metadata, normalizer)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        if not self._update_incremental:
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern,
                                                                                                            self.X,
                                                                                                            self.likelihood,
                                                                                                            self.Y_normalized,
                                                                                                            self.mean_function,
                                                                                                            self.Y_metadata)
            if self._calculate_gradients: # switch for updating gradients
                self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
                self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
                if self.mean_function is not None:
                    self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)
            self.kern.update_non_incremental(self.X)
        else:
            # print("performing incremental update")
            self.kern.update_incremental(self.X)
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.incremental_inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)

            if self._calculate_gradients: # switch for updating gradients
                self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
                if hasattr(self.kern, 'update_gradients_incremental'):
                    self.kern.update_gradients_incremental(self.grad_dict['dL_dK'], self.X)
                else:
                    self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
                if self.mean_function is not None:
                    self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

            self._update_incremental = False

    def append_XY(self, X, Y):
        """
        The real functionality is implemented in self._appendXY.
        This is to make sure, that on the first call of either append_XY or set_XY we remove the fake data point
        """
        super().set_XY(X, Y)
        # remap functions
        self.append_XY = self._append_XY
        self.set_XY = super().set_XY

    def set_XY(self, X,Y):
        """
        The real functionality is implemented in super().set_XY
        This is to make sure, that on the first call of either append_XY or set_XY we remove the fake data point
        """
        super().set_XY(X, Y)
        # remap functions
        self.append_XY = self._append_XY
        self.set_XY = super().set_XY


    def _append_XY(self, X, Y):
        """
        append X,Y to current data, perform incremental model update.
        needs an inference method which supports incremental updates
        Can only add a single data point for now
        :param X: X to be appended
        :param Y: Y to be appended
        """
        X = np.concatenate((self.X, X))
        Y = np.concatenate((self.Y, Y))
        self._update_incremental = True
        self.set_XY(X, Y)

class GPRegression(GP):
    """
    Mimics GPy.models.GPRegression, but with our own GP base class
    """

    def __init__(self, input_dim, kernel, mean_function=None, noise_var=1., Y_metadata=None, normalizer=False, calculate_gradients=True):
        likelihood = GPy.likelihoods.Gaussian(variance=noise_var)

        super().__init__(input_dim, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function, calculate_gradients=calculate_gradients)


class HeteroscedasticGaussian(GPy.likelihoods.Gaussian):
    """ Similar to GPy.likelihoods.Gaussian, but stores the variance in Y_metadata, which is easier to modifiy than paramz paramters."""
    # We cannot do inference gradients since we don't store variance as Paramz
    # def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
    #     return dL_dKdiag #[Y_metadata['variance']]

    def gaussian_variance(self, Y_metadata=None):
        return Y_metadata['variance'].flatten()

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        _s = Y_metadata['variance'].flatten()
        if full_cov:
            if var.ndim == 2:
                var += np.eye(var.shape[0])*_s
            if var.ndim == 3:
                var += np.atleast_3d(np.eye(var.shape[0])*_s)
        else:
            var += _s
        return mu, var

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        _s = Y_metadata['output_index'].flatten()
        return  [stats.norm.ppf(q/100.)*np.sqrt(var + _s) + mu for q in quantiles]

class GPHeteroscedasticRegression(GP):
    def __init__(self, input_dim, kernel=None, calculate_gradients=True, normalizer=False):

        self.S = np.ones(1).reshape(1,-1)
        Y_metadata = {'variance': self.S}  # continue with the hack of the 'empty' gp in base class

        # Likelihood
        likelihood = HeteroscedasticGaussian()

        super().__init__(input_dim, kernel, likelihood, Y_metadata=Y_metadata, normalizer=normalizer, calculate_gradients=calculate_gradients)


    def set_XY(self, X,Y, S):
        """

        Args:
            X:
            Y:
            S: standard deviation

        Returns:

        """
        self.S = S
        self.Y_metadata['variance'] = S*S
        super().set_XY(X,Y)

    def append_XY(self, X, Y, S):
        """

        Args:
            X:
            Y:
            S: standard deviation

        Returns:

        """
        self.set_XY(X, Y, S) # super class remaps this methods to _append_XY after first call, see documentation there

    def _append_XY(self, X, Y, S):
        self.S = np.concatenate((self.S, S))
        self.Y_metadata['variance'] = np.concatenate((self.Y_metadata['variance'], S*S))
        super()._append_XY(X,Y)




class ExactGaussianInferenceIncremental(GPy.inference.latent_function_inference.ExactGaussianInference):
    def __init__(self, *args, **kwargs):
        super(ExactGaussianInferenceIncremental, self).__init__(*args, **kwargs)
        self._incremental_update = False
        self._old_LW = None
        self._old_X = None
        self._old_Wi = None
        self._K = None

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None,
                  Z_tilde=None):
        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        posterior = super(ExactGaussianInferenceIncremental, self).inference(kern, X, likelihood, Y, mean_function,
                                                                        Y_metadata, K, variance, Z_tilde)
        self._old_LW = posterior[0].woodbury_chol
        self._K = kern.K(X).copy()
        self._old_Wi, _ = dpotri(self._old_LW, lower=1)
        # diag.add(self._K, variance + 1e-8)

        return posterior

    def incremental_inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None,
                  Z_tilde=None):

        # do incremental update
        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y - m


        # K_tmp = kern.K(X, X[-1:])
        K_inc = kern._K[:-1,-1]
        K_inc2 = kern._K[-1:,-1]
        # self._K = np.block([[self._K, K_inc], [K_inc.T, K_inc2]])

        # Ky = K.copy()
        jitter = variance[-1] + 1e-8 # variance can be given for each point individually, in which case we just take the last point
        # diag.add(Ky, jitter)

        # LW_old = self._old_posterior.woodbury_chol

        Wi, LW, LWi, W_logdet = pdinv_inc(self._old_LW, K_inc, K_inc2 + jitter, self._old_Wi)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal = 0.5 * (-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        self._old_LW = LW
        self._old_Wi = Wi
        posterior = Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K)

        # TODO add logdet to posterior ?
        return posterior, log_marginal, {'dL_dK': dL_dK, 'dL_dthetaL': dL_dthetaL, 'dL_dm': alpha}


def pdinv_inc(L_old, A_inc, A_inc2, Ai_old, *args):
    """
    similar to pdinv, but uses old choleski decompositon to compute new
    as proposed in https://github.com/SheffieldML/GPy/issues/464#issuecomment-285500122

    :rval Ai: the inverse of A
    :rtype Ai: np.ndarray
    :rval L: the Cholesky decomposition of A
    :rtype L: np.ndarray
    :rval Li: the Cholesky decomposition of Ai
    :rtype Li: np.ndarray (set to None for now, because not needed)
    :rval logdet: the log of the determinant of A
    :rtype logdet: float64
    """
    """ 
    """
    # A_inc = A_inc.reshape(-1)
    u = dtrtrs(L_old, A_inc, lower=1)[0]
    v = np.sqrt(A_inc2 - np.sum(u*u)).reshape(1)
    z = np.zeros((A_inc.shape[0],1))
    L = np.asfortranarray(np.block([[L_old, z], [u, v]]))
    logdet = 2. * np.sum(np.log(np.diag(L)))

    # _Ai, _ = dpotri(L, lower=1)  # consider also incrementally updating this

    # incrementally update the inverse
    alpha = Ai_old.dot(A_inc).reshape((-1,1))
    # print('alpha', alpha)
    gamma = alpha.dot(alpha.T)
    # print('gamma', gamma)
    beta = 1/(A_inc2 - alpha.T.dot(A_inc)).reshape(-1)
    beta_alpha = -beta*alpha
    # print('beta', beta)
    Ai = np.block([[Ai_old + beta * gamma, beta_alpha], [beta_alpha.T, beta]])
    # print(Ai -_Ai)

    symmetrify(Ai)

    return Ai, L, None, logdet  # could also return Li as in pdinv, but we don't need this right now

def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation (https://stackoverflow.com/a/31075641) """
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls),{})

# Felix
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