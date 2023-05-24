import numpy as np
import torch
from febo.utils import locate, get_logger
from febo.utils.config import ConfigField, Config, assign_config, Configurable
from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig

from stpy.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
logger = get_logger('model')




class StandaloneGPConfig(ModelConfig):

    kernels = ConfigField([('ard', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True, 'groups':None})])
    noise_var = ConfigField(0.1)
    calculate_gradients = ConfigField(True, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(True)
    optimize_var = ConfigField(True)
    bias = ConfigField(0)



@assign_config(StandaloneGPConfig)
class StandaloneGP(ConfidenceBoundModel):

    def __init__(self, domain):
        """
        Args:
            d (int): input dimension
        """
        self.domain = domain
        self.d = domain.d
        self.fit = False
        self.s = self.config.noise_var
        self.kernel_name = self.config.kernels[0][0]
        self.gamma = self.config.kernels[0][1]['lengthscale']

        if self.config.kernels[0][1]['groups'] is None:
            self.groups = self.config.kernels[0][1]['groups']
        else:
            self.groups = eval(self.config.kernels[0][1]['groups'])

        kernel = KernelFunction(kernel_name=self.kernel_name, gamma=torch.ones(self.d, dtype=torch.float64) * self.gamma, groups=self.groups)
        self.gp = GaussianProcess(kernel_custom=kernel, s=self.s, d=self.d)
        self._beta_cached = 2

        self.X = None
        self.Y = None

    def mean(self, X):
        """
        Calculate predicted mean at X.

        Args:
            X: input narray of shape (N,d)

        Returns:
            Predicted mean, narray of shape (N,1)

        """
        X = torch.from_numpy(X.reshape(-1,self.d))
        mean,_ = self.gp.mean_var(X)
        return mean.numpy()

    def var(self, X):
        """
        Calculate predicted variance at X

        Args:
            X: input narray of shape (N,d)

        Returns:
            Predicted variance, narray of shape (N,1)

        """
        X = torch.from_numpy(X.reshape(-1,self.d))
        mean,var = self.gp.mean_var(X)
        return var.numpy()


    def _beta(self):
        return self._beta_cached

    def mean_var(self, X):
        """

        Calculate predicted mean and variance at X

        Args:
            X: input narray of shape (N,d)

        Returns:
            (mean, var) Predicted mean and variance, each narray of shape (N,1)

        """
        X = torch.from_numpy(X.reshape(-1,self.d))
        mean,var = self.gp.mean_var(X)
        return (mean.numpy(),var.numpy())

    def sample(self, X=None):
        """
        Returns a sample form the posterior. It should return a function ``def my_sample(X):``

        Args:
            X: if specified, the sample function is only valid for points X

        Returns (function):

        """
        def sampler(X):
            X = torch.from_numpy(X.reshape(-1, self.d))
            f = self.gp.sample(X).numpy()
            return f

        # def sampler_coord(X):
        #     X = torch.from_numpy(X.reshape(-1, self.d))
        #     x,val = self.gp.sample_iteratively_max(X, multistart = 20, minimizer = "coordinate-wise", grid = 100).numpy()
        #     return (x,val )
        return sampler



    def fit_gp(self):
        self.gp.fit_gp(self.X,self.Y)
        self.fit = True
    def add_data(self, X, Y):
        """
        Add data to the model.

        Args:
            X: input narray of shape (N,d)
            Y: observation narray of shape (N,1)

        """
        X = torch.from_numpy(X.reshape(-1,self.d))
        Y = torch.from_numpy(Y.reshape(-1,1))

        if self.X is None:
            self.X = X
            self.Y = Y
        else:
            self.X = torch.cat((self.X,X), dim=0)
            self.Y = torch.cat((self.Y, Y), dim=0)
        self.fit_gp()

    def info(self):
        return self.gp.description()

    @property
    def requires_std(self):
        return False
