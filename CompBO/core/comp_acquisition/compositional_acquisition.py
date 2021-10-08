from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Callable

import numpy as np
import torch
from botorch.utils import draw_sobol_normal_samples
from torch import Tensor


class CompositionalAcquisition(ABC):
    """
    Abstract class for Compositional Acquisition Functions: alpha(x) = f(E[g(x)]). Subclass of CompositionalAcquisition
    shall also be subclass of `botorch` AcquisitionFunction
    """

    def __init__(self, fixed_z: bool, K_g: int, m: int, approx: bool = False):
        """

        Args:
            fixed_z: whether to use fixed z samples across optimization steps (set to `False` for memory efficient (ME))
            K_g: number of inner samples used at each optimization step
            m: number of z samples considered to build `g` (should be equal to `K_g` for `ME` version)
            approx: use qMC samples
        """
        self.base_samples_z = None
        self.fixed_z = fixed_z
        self.sampler.resample = not fixed_z

        self.K_g = K_g
        self.oracle_g = self.inner_g_oracle
        self.m = m
        self.approx = approx

    @abstractmethod
    def outer_f(self, Y: Tensor) -> Tensor:
        """ Deterministic outer function `f`"""
        pass

    @abstractmethod
    def inner_g_expected(self, X: Tensor) -> Tensor:
        """ Inner function `x -> E_w[g_w(x)]` """
        pass

    @abstractmethod
    def inner_g_oracle(self, X: Tensor, custom_z_filter: Optional[Tensor] = None) -> Tensor:
        """
        Oracle for inner function `g_w`
        Args:
            X: tensor of inputs
            custom_z_filter: if specified, use this tensor to select the `z` samples

        Returns:
            Empirical mean of the g_w(x)
        """
        pass

    @abstractmethod
    def nested_eval(self, X: Tensor, **kwargs) -> Tensor:
        """ Evaluation of acquisition function in nested form """
        pass

    def opt_forward(self, X: Tensor, Y: Tensor, eval_J: Optional[bool] = True, new_samples: bool = True) \
            -> Union[Tuple[Callable, Tensor, Tensor], Tuple[Callable, Tensor]]:
        """ Evaluate acquisition function on the candidate set `X` and auxilliary Y.

        Args:
            X: A `t-batch x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
            Y: A `t-batch x q x num_samples Tensor (same dimension as g(X) in C-Adam)
            eval_J: if True, compute f(g(x)) and return it as variable `loss`. But as this value can be obtained from
                CAdam  optimizer it can be computationally cheaper to evaluate J after optimization step

        Returns:
            g (Callable): function g: Tensor (`batch_shape x q x d`) -> Tensor (`batch_shape x q x num_samples`)
            f_y (Tensor): a `batch_shape'`-dim Tensor of Expected Improvement values
            loss (Optional[Tensor]): a `batch_shape'`-dim Tensor of Expected Improvement values
        """
        if new_samples:
            self.gen_z_ind_samples(device=X.device)
        Y_aux = Y[..., self.z_filter]
        g = self.oracle_g
        f_Y = self.outer_f(Y_aux)

        if eval_J:
            with torch.no_grad():
                g_X: Tensor = g(X)
                loss = self.outer_f(g_X)
            return g, f_Y, loss
        return g, f_Y

    def z_samples(self, *size, dtype, device=None) -> Tensor:
        """ Get the `z` normal samples"""

        if hasattr(self, 'approx') and self.approx:
            return self.z_sobol_samples(*size, dtype=dtype, device=device)
        if self.base_samples_z is None or not self.fixed_z or self.base_samples_z.shape != (*size,):
            self.base_samples_z = torch.randn(*size, dtype=dtype, device=device)
        return self.base_samples_z

    def z_sobol_samples(self, *size, dtype, device=None) -> Tensor:
        """ Get `z` samples from Sobol sequence """
        if self.base_samples_z is None or not self.fixed_z or self.base_samples_z.shape != (*size,):
            assert len(size) == 2, size
            self.base_samples_z = draw_sobol_normal_samples(d=size[0], n=size[1], dtype=dtype, device=device).permute(1,
                                                                                                                      0)
        return self.base_samples_z

    def gen_z_ind_samples(self, device=None) -> None:
        """ Get indices of the `z` samples to consider when calling `g` oracle """
        if hasattr(self, 'approx') and self.approx:
            # get adjacent samples
            self.z_filter = torch.zeros(self.m, dtype=bool)
            self.z_filter[:self.K_g] = 1
            self.z_filter = torch.roll(self.z_filter, np.random.randint(self.get_m()))
        else:
            z_inds = torch.randint(0, self.get_m(), size=(self.K_g,))
            self.z_filter = torch.zeros(self.m, dtype=bool)
            self.z_filter[z_inds] = 1
        self.Kt_g = self.z_filter.sum().item()
        self.z_filter = self.z_filter.to(device=device)
        assert self.Kt_g <= self.K_g, (self.K_g, self.Kt_g)  # sampling with replacement

    def set_z_ind_samples(self, z_inds, device=None) -> None:
        """ Set values of indices of the `z` samples to consider when calling `g` oracle """
        self.z_filter = torch.zeros(self.m, dtype=bool, device=device)
        self.z_filter[z_inds] = 1
        self.Kt_g = self.z_filter.sum().item()

    def get_m(self) -> int:
        """ Return the number of MC samples used in total """
        return self.m
