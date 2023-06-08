# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.special import jv
from scipy.integrate import quad
from scipy.stats import rv_discrete
import sobol_seq
from scipy.stats import multivariate_normal


## Global optimization benchmark functions
# Ackley function
# https://www.sfu.ca/~ssurjano/ackley.html
def ackley(x_ori):
    x = x_ori.copy()
    x -= 0.5
    x *= 10
    num = float(len(x[0]))
    y = []
    for n in x:
        firstSum = 0
        secondSum = 0
        for c in n:
            firstSum += c**2
            secondSum += np.cos(2.0*np.pi*c)
        y.append(-20.0*np.exp(-0.2*np.sqrt(firstSum/num)) - np.exp(secondSum/num) + 20 + np.e)
	
    return -1*(np.array(y)).reshape(-1,1)/10 + 1

def ackely_max_min(x,dim):
    
    y = ackley(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = ackley(max_pos)
    return max_pos, max, None, None

def ackley_max_min_var(x,dim,t,s):
    max_pos, max, min_pos, min = ackely_max_min(x,dim)
    max_pos = max_pos + t
    # min_pos = min_pos + t

    return max_pos, s * max, min_pos, None

def ackley_var(x,t,s):
    x_new = x.copy()
    dim = len(x[0])
    # apply translation
    t_range = np.array([[0, 1]]*dim)
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * ackley(x_new)
# POWELL function
# https://www.sfu.ca/~ssurjano/powell.html
def POWELL(x_ori):
    x = x_ori.copy()
    dim = len(x[0])
    x -= 0.5
    x *= 8
    s = np.zeros(x.shape[0])
    for i in range(dim//4):
        term1 = (x[:,int(4*(i+1)-3)] + 10*x[:,int(4*(i+1)-2)])**2
        term2 = 5*(x[:,int(4*(i+1)-1)] - x[:,int(4*(i+1))])**2
        term3 = (x[:,int(4*(i+1)-2)] - 2* x[:,int(4*(i+1)-1)])**4
        term4 = 10*(x[:,int(4*(i+1)-3)] - x[:,int(4*(i+1))])**4
        s += term1 + term2 + term3 + term4
    return s.reshape(-1,1) / 100000 + 1


def POWELL_max_min(x,dim):
    y = POWELL(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = POWELL(max_pos)
    min_pos = np.array([x[np.argmin(y)]])
    min = POWELL(min_pos)
    return max_pos, max, min_pos, min

def POWELL_max_min_var(x,dim,t,s):
    max_pos, max, min_pos, min = POWELL_max_min(x,dim)
    max_pos = max_pos + t
    min_pos = None

    return max_pos, s * max, None, None
def POWELL_var(x,t,s):
    x_new = x.copy()
    dim = len(x[0])
    x_new = x_new - t

    return s * POWELL(x_new)


# DIXON-PRICE
# https://www.sfu.ca/~ssurjano/dixonpr.html
def DIXON_PRICE(x_ori):
    x = x_ori.copy()
    x -= 0.5
    x *= 6
    x1 = x[:,0]
    dim = len(x[0])
    term1 = (x1-1)**2
    s = np.zeros((x.shape[0]))
    for i in range(1,dim):
        xi = x[:,i]
        xold = x[:,i-1]
        new = (i+1) * (2*xi**2 - xold)**2
        s += new
    return - ( (term1 + s).reshape(-1,1)) / 10000 + 1
def DIXON_PRICE_max_min(x,dim):
    # x = sobol_seq.i4_sobol_generate(dim,1000000)
    y = DIXON_PRICE(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = DIXON_PRICE(max_pos)
    min_pos = None
    min = None
    return max_pos, max, min_pos, min

def DIXON_PRICE_max_min_var(x,dim,t,s):
    max_pos, max, min_pos, min = DIXON_PRICE_max_min(x,dim)
    max_pos = max_pos + t
    min_pos = None

    return max_pos, s * max, None, None
def DIXON_PRICE_var(x,t,s):
    x_new = x.copy()
    dim = len(x[0])
    # apply translation
    x_new = x_new - t

    return s * DIXON_PRICE(x_new)   

# STYBLINSKI-TANG function
# https://www.sfu.ca/~ssurjano/stybtang.html
def STYBLINSKI_TANG(x_ori):
    x = x_ori.copy()
    x -= 0.5
    x *= 10 # move the x domain to [-5,5]
    dim = len(x[0])
    s = np.zeros((x.shape[0]))
    for i in range(dim):
        xi = x[:,i]
        new = xi**4 - 16*(xi**2) + 5*xi
        s += new
    return -np.array(s/2).reshape(-1,1)/(40*dim)
def STYBLINSKI_TANG_max_min(x,dim):
    # x = sobol_seq.i4_sobol_generate(dim,1000000)
    y = STYBLINSKI_TANG(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = STYBLINSKI_TANG(max_pos)
    min_pos = None
    min = None
    return max_pos, max, min_pos, min

def STYBLINSKI_TANG_max_min_var(x,dim,t,s):
    max_pos, max, min_pos, min = STYBLINSKI_TANG_max_min(x,dim)
    max_pos = max_pos + t
    min_pos = None

    return max_pos, s * max, None, None
def STYBLINSKI_TANG_var(x,t,s):
    x_new = x.copy()
    dim = len(x[0])
    # apply translation
    x_new = x_new - t

    return s * STYBLINSKI_TANG(x_new)

# GRIEWANK function
# https://www.sfu.ca/~ssurjano/griewank.html
def GRIEWANK(x_ori):
    x = x_ori.copy()
    x -= 0.5
    x *= 1200 # move the x domain to [-600,600]
    dim = len(x[0])
    total = np.zeros(x.shape[0])
    prod = 1
    for i in range(dim):
        xi = x[:,i]
        total += xi**2/4000
        prod = prod * np.cos(xi/np.sqrt(i+1))
    return -np.array(total - prod + 1).reshape(-1,1) / 200 + 1

def GRIEWANK_max_min(x,dim):
    y = GRIEWANK(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = GRIEWANK(max_pos)
    min_pos = None
    min = None
    return max_pos, max, min_pos, min

def GRIEWANK_max_min_var(x,dim,t,s):
    max_pos, max, min_pos, min = GRIEWANK_max_min(x,dim)
    max_pos = max_pos + t
    min_pos = None

    return max_pos, s * max, None, None
def GRIEWANK_var(x,t,s):
    x_new = x.copy()
    dim = len(x[0])
    # apply translation
    x_new = x_new - t

    return s * GRIEWANK(x_new)


# the egg holder function 2D
# https://www.sfu.ca/~ssurjano/egg.html
def Eggholder(x_ori):
    x = x_ori.copy()
    # scale up sobol sequence
    x -= 0.5
    x *= 1024
    # max position
    x1 = x[:,0]
    x2 = x[:,1]
    return -( -(x2+47)*np.sin(np.sqrt(np.abs(x2+x1/2+47))) - x1*np.sin(np.sqrt(np.abs(x1-(x2+47)))) ).reshape(-1,1)/1000
def Eggholder_max_min(x):
    y = Eggholder(x)
    max_pos = np.array([x[np.argmax(y)]])
    max = Eggholder(max_pos)
    min_pos = None
    min = None
    return max_pos, max, min_pos, min

def Eggholder_max_min_var(x,t,s):
    max_pos, max, min_pos, min = Eggholder_max_min(x)
    max_pos = max_pos + t
    min_pos = None

    return max_pos, s * max, None, None

def Eggholder_var(x,t,s):
    x_new = x.copy()
    x_new = x_new - t

    return s * Eggholder(x_new)

# modify from MetaBO
class SparseSpectrumGP:
    """
    Implements the sparse spectrum approximation of a GP following the predictive
    entropy search paper.

    Note: This approximation assumes that we use a GP with squared exponential kernel.
    """

    def __init__(self, input_dim, seed, noise_var=1.0, length_scale=1.0, signal_var=1.0, n_features=100, kernel="RBF",periods=[0.3,0.6]):
        self.seed = seed
        self.rng = np.random.RandomState()
        self.rng.seed(self.seed)

        self.input_dim = input_dim
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.n_features = n_features

        self.period = periods # for Expss kernel
        
        self.min_tol = 0
        self.xf_limit = 200 # cuf-off point for the spectrum
        self.N = 100  # number of points in the spectrum

        self.kernel = kernel
        self.phi = self._compute_phi()
        self.jitter = 1e-10


        self.X = None
        self.Y = None

        # Statistics of the weights that give us random function samples
        # f(x) ~ phi(x).T @ theta, theta ~ N(theta_mu, theta_var)
        self.theta_mu = None
        self.theta_var = None


    def train(self, X, Y, n_samples):
        """
        Pre-compute all necessary variables for efficient prediction and sampling.
        """
        self.X = X
        self.Y = Y

        phi_train = self.phi(X)
        a = phi_train.T @ phi_train + self.noise_var * np.eye(self.n_features) # phi.T @ phi + diagonal_noise
        a_inv = np.linalg.inv(a) # (phi.T @ phi)^-1
        self.theta_mu = a_inv @ phi_train.T @ Y # phi^-1 @ Y
        self.theta_var = self.noise_var * a_inv # noise * a

        # Generate handle to n_samples function samples that can be evaluated at x.
        var = self.theta_var + self.jitter * np.eye(self.theta_var.shape[0])
        var = (var + var.T) / 2
        chol = np.linalg.cholesky(var) # chol^-1 @ chol^-T = var
        self.theta_samples = self.theta_mu + chol @ self.rng.randn(self.n_features, n_samples) 

    def predict(self, Xs, full_variance=False):
        raise NotImplementedError

    def sample_posterior(self, Xs):
        """
        Generate n_samples function samples from GP posterior at points Xs.
        """
        h = self.sample_posterior_handle
        return h(Xs)

    def sample_posterior_handle(self, x):
        x = np.atleast_2d(x).T if x.ndim == 1 else x
        return self.theta_samples.T @ self.phi(x).T

    def _compute_phi(self):
        """
        Compute random features.
        """
        if self.kernel == "RBF":
            w = self.rng.randn(self.n_features, self.input_dim) / self.length_scale 
        elif self.kernel == "Matern32":
            w = self.rng.standard_t(3, (self.n_features, self.input_dim)) / self.length_scale
        elif self.kernel == "Matern52":
            w = self.rng.standard_t(5, (self.n_features, self.input_dim)) / self.length_scale
        elif self.kernel == "SM":
            period = self.rng.choice(self.period)
            m = np.ones(self.input_dim)*1/period
            v = np.ones(self.input_dim)*(1/((self.length_scale)**2))
            w = self.rng.normal(m, v, (self.n_features, self.input_dim))

        b = self.rng.uniform(0, 2 * np.pi, size=self.n_features)
        return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(x @ w.T + b)

def HPO(x, data, index=None):
    # breakpoint()
    if index is not None:
        ret = data["accs"][index]
        return np.array(ret).reshape(-1, 1)
    else:
        X = get_HPO_domain(data)
        try:
            idx = np.asscalar(np.where(np.all(x == X, axis=1))[0][0])
            # idx = np.asscalar(np.where(np.all(np.isclose(x, X), axis=1))[0][0])
        except:
            print('[HPO] except')
            print(x)
            print(np.all(x == X, axis=1))
            print(np.where(np.all(x == X, axis=1)))
            print((np.where(np.all(x == X, axis=1))[0]))
            exit(10)
        ret = data["accs"][idx]
        return np.array(ret).reshape(-1,1)

def get_HPO_domain(data):
    return np.array(data["domain"])

def HPO_max_min(data):
    X = get_HPO_domain(data)
    idx_max = np.argmax(data["accs"])
    idx_min = np.argmin(data["accs"])

    return X[idx_max], data["accs"][idx_max], X[idx_min], data["accs"][idx_min]


def get_Antigen_domain(data):
    return np.array(data["domain"])

def Antigen_max_min(data):
    X = get_Antigen_domain(data)
    idx_max = np.argmax(data["accs"])
    idx_min = np.argmin(data["accs"])

    return X[idx_max], data["accs"][idx_max], X[idx_min], data["accs"][idx_min]

def Antigen(x, data, index=None):
    # breakpoint()
    if index is not None:
        ret = data["accs"][index]
        return np.array(ret).reshape(-1, 1)
    else:
        X = get_Antigen_domain(data)
        try:
            idx = np.asscalar(np.where(np.all(x == X, axis=1))[0][0])
            # idx = np.asscalar(np.where(np.all(np.isclose(x, X), axis=1))[0][0])
        except:
            print('FAILED to find x in data.')
            print(x)
            print(np.all(x == X, axis=1))
            print(np.where(np.all(x == X, axis=1)))
            print((np.where(np.all(x == X, axis=1))[0]))
            exit(10)
        ret = data["accs"][idx]
        return np.array(ret).reshape(-1,1)
