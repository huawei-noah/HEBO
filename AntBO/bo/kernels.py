# Implementation of various kernels

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from gpytorch.kernels.cosine_kernel import CosineKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from torch import Tensor


def wrap(x1, x2, integer_dims):
    """The wrapping transformation for integer dimensions according to Garrido-Merchán and Hernández-Lobato (2020)."""
    if integer_dims is not None:
        for i in integer_dims:
            x1[:, i] = torch.round(x1[:, i])
            x2[:, i] = torch.round(x2[:, i])
    return x1, x2


class WrappedMatern(MaternKernel):
    """Matern kernels wrapped integer type of inputs according to
    Garrido-Merchán and Hernández-Lobato in
    "Dealing with Categorical and Integer-valued Variables in Bayesian Optimization with Gaussian Processes"

    Note: we deal with the categorical-valued variables using the kernels specifically used to deal with
    categorical variables (instead of the one-hot transformation).
    """

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedMatern, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class WrappedRBF(RBFKernel, WrappedMatern):
    """Similar to above, but applied to RBF."""

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedRBF, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        diff = x1[:, None] - x2[None, :]
        diff[torch.abs(diff) > 1e-5] = 1
        diff1 = torch.logical_not(diff).float()

        def rbf(d, ard):
            if ard:
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class OrdinalKernel(Kernel):
    """
    The ordinal version of TransformedCategorical2 kernel (replace the Kronecker delta with
    the distance metric).
    config: the number of vertices per dimension
    """

    def __init__(self, config, **kwargs):
        super(OrdinalKernel, self).__init__(has_lengthscale=True, **kwargs)
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config).view(-1)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # expected x1 and x2 are of shape N x D respectively
        diff = (x1[:, None] - x2[None, :]) / self.config
        dist = 1. - torch.abs(diff)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.exp(
                torch.sum(
                    dist * self.lengthscale, dim=-1
                ) / torch.sum(self.lengthscale)
            )
        else:
            k_cat = torch.exp(
                self.lengthscale * torch.sum(dist, dim=-1) / x1.shape[1]
            )
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class FastStringKernel(Kernel):
    """
    Code based on https://github.com/beckdaniel/flakes
    We make following changes
    1) provide kernel normalization to make meaningful comparissons between strings of different lengths
    2) changed structure and conventions to match our Tree kernel implemenentation
    3) simplified to only allow one-hot encoidng of alphabet (i.e remove support for pre-trained embeddings)
    4) a collection of performance tweaks to improve vectorization
    """

    def __init__(self, seq_length: int, alphabet_size: int, gap_decay=.5, match_decay=.8,
                 max_subsequence_length: int = 3, normalize=True, **kwargs):
        super(FastStringKernel, self).__init__(has_lengthscale=False, **kwargs)

        self.register_parameter(name='match_decay', parameter=torch.nn.Parameter(torch.tensor(match_decay)))
        self.register_parameter(name='gap_decay', parameter=torch.nn.Parameter(torch.tensor(gap_decay)))
        self.register_constraint("gap_decay", Interval(0, 1))
        self.register_constraint("match_decay", Interval(0, 1))
        self.max_subsequence_length = max_subsequence_length

        # store additional kernel parameters
        self.maxlen = seq_length
        self.alphabet_size = alphabet_size
        self.normalize = normalize

        self.tril = torch.triu(torch.ones((self.maxlen, self.maxlen), dtype=torch.double), diagonal=1).to(
            kwargs['device'])
        self.exp = torch.ones(self.maxlen, self.maxlen, dtype=int).to(kwargs['device'])
        for i in range(self.maxlen - 1):
            self.exp[i, i + 1:] = torch.arange(self.maxlen - i - 1)

        self.symmetric = None
        self.D = None

    @staticmethod
    def K_diag(self, x: torch.tensor) -> torch.tensor:
        r"""
        The diagonal elements of the string kernel are always unity (due to normalisation)
        """
        return torch.ones(x.shape[:-1], dtype=torch.double)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        r"""
        Vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

        # Turn our inputs into lists of integers using one-hot embedding
        # first split up strings and pad to fixed length and prep for gpu
        # pad until all have length of self.maxlen
        if diag:
            raise ValueError()
        if x2 is None:
            x2 = x1
            self.symmetric = True
        else:
            self.symmetric = False
        # keep track of original input sizes
        x1_shape = x1.shape[0]
        x2_shape = x2.shape[0]

        # prep the decay tensor D
        self.D = self._precalc().to(x1)

        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        x1 = torch.nn.functional.one_hot(x1.to(int), self.alphabet_size).to(x1)
        x2 = torch.nn.functional.one_hot(x2.to(int), self.alphabet_size).to(x2)

        # get indicies of all possible pairings from X and x2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indicies_2, indicies_1 = torch.meshgrid(torch.arange(0, x2.shape[0]), torch.arange(0, x1.shape[0]))
        indicies = torch.cat([torch.reshape(indicies_1.T, (-1, 1)), torch.reshape(indicies_2.T, (-1, 1))], axis=1)

        # if symmetric then only calc upper matrix (fill in rest later)
        if self.symmetric:
            indicies = indicies[indicies[:, 1] >= indicies[:, 0]]

        x1_full = torch.repeat_interleave(x1.unsqueeze(0), len(indicies), dim=0)[
            np.arange(len(indicies)), indicies[:, 0]]
        x2_full = torch.repeat_interleave(x2.unsqueeze(0), len(indicies), dim=0)[
            np.arange(len(indicies)), indicies[:, 1]]

        if not self.symmetric:
            # also need to calculate some extra kernel evals for the normalization terms
            x1_full = torch.cat([x1_full, x1, x2], 0)
            x2_full = torch.cat([x2_full, x1, x2], 0)

        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = torch.matmul(x1_full, torch.transpose(x2_full, 1, 2))

        # store squared match coef
        match_sq = self.match_decay ** 2

        Kp = torch.ones(*[S.shape[0], self.maxlen, self.maxlen]).to(S)

        # do all remaining steps
        for i in torch.arange(self.max_subsequence_length - 1):
            Kp = torch.multiply(S, Kp)
            Kp = match_sq * Kp
            Kp = torch.matmul(Kp, self.D)
            Kp = torch.matmul(self.D.T, Kp)

        # final kernel calc
        Kp = torch.multiply(S, Kp)
        k = Kp.sum((-2, -1)).unsqueeze(1) * match_sq

        # put results into the right places in the gram matrix and normalize
        if self.symmetric:
            # if symmetric then only put in top triangle (inc diag)
            mask = torch.triu(torch.ones((x1_shape, x2_shape)), 0).to(S)
            non_zero = mask > 0
            k_results = torch.zeros((x1_shape, x2_shape)).to(S)
            k_results[non_zero] = k.squeeze()
            # add in mising elements (lower diagonal)
            k_results = k_results + k_results.T - torch.diag(k_results.diag())

            # normalise
            X_diag_Ks = torch.diag(k_results)
            norm = torch.matmul(X_diag_Ks[:, None], X_diag_Ks[None, :])
            k_results = torch.divide(k_results, torch.sqrt(norm))
        else:
            # otherwise can just reshape into gram matrix
            # but first take extra kernel calcs off end of k

            # COULD SPEED THIS UP FOR PREDICTIONS, AS MANY NORM TERMS ALREADY IN GRAM

            X_diag_Ks = k[x1_shape * x2_shape:x1_shape * x2_shape + x1_shape].flatten()

            x2_diag_Ks = k[-x2_shape:].flatten()

            k = k[0:x1_shape * x2_shape]
            k_results = k.reshape(x1_shape, x2_shape)

            # normalise
            norm = torch.matmul(X_diag_Ks[:, None], x2_diag_Ks[None, :])
            k_results = torch.divide(k_results, torch.sqrt(norm))

        return k_results

    def _precalc(self):
        r"""
        Precalc D matrix as required for kernel calcs
        following notation from Beck (2017)
        """
        return torch.pow(self.gap_decay * self.tril, self.exp)


class BERTWarpCosine(CosineKernel):
    """Applied to Cosine."""

    def __init__(self, **kwargs):
        super(BERTWarpCosine, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        return super().forward(x1, x2, diag=diag, **params)


class BERTWarpRBF(RBFKernel):
    """Similar to above, but applied to RBF."""

    def __init__(self, **kwargs):
        super(BERTWarpRBF, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        return super().forward(x1, x2, diag=diag, **params)


if __name__ == '__main__':
    # Test whether the ordinal kernel is doing ok
    import numpy as np
    import matplotlib.pyplot as plt

    x1_ = torch.tensor([[13., 4.],
                       [43., 15.],
                       [32., 19.],
                       [41., 9.],
                       [47., 44.],
                       [48., 21.],
                       [15., 24.],
                       [20., 13.],
                       [36., 46.],
                       [19., 17.],
                       [35., 6.],
                       [39., 50.],
                       [24., 10.],
                       [45., 18.],
                       [29., 3.],
                       [17., 27.],
                       [25., 16.],
                       [37., 29.],
                       [16., 2.],
                       [3., 38.]])

    o = OrdinalKernel(config=[51, 51])
    o.lengthscale = 1.
    K = o.forward(x1_, x1_).detach().numpy()
    plt.imshow(K)
    plt.colorbar()
    plt.show()
