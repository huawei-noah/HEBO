from typing import Type, List, Optional

import torch
from botorch import test_functions, acquisition
from botorch.acquisition import MCAcquisitionFunction
from botorch.test_functions import SyntheticTestFunction
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, Kernel
from gpytorch.priors import GammaPrior
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer

import custom_optimizer
from core import comp_acquisition
from core.es import evolution_opt


def query_test_func(test_func_name: str, input_dim: int, negate: bool, l_bound_offset: Optional[float] = None,
                    u_bound_offset: Optional[float] = None) -> SyntheticTestFunction:
    """ Get instance of `botorch` synthetic function

     Args:
         test_func_name: name of the synthetic function `f` (Levy, DixonPrice...)
         input_dim: input dimension
         negate: whether to con consider `-f` instead of `f`
         l_bound_offset: offset added to the lower bound of the hypercube domain of `f`
         u_bound_offset: offset added to the upper bound of the hypercube domain of `f`
     """
    func = getattr(test_functions, test_func_name)
    if hasattr(func, "dim"):
        if func.dim != input_dim:
            raise ValueError(f"{test_func_name} does not allow input dimension {input_dim}, only {func.dim}")
        f = func(negate=negate)
    else:
        f = func(dim=input_dim, negate=negate)
    if l_bound_offset is None:
        l_bound_offset = 0
    if u_bound_offset is None:
        u_bound_offset = 0
    if u_bound_offset != 0 or l_bound_offset != 0:
        print(f'Former bounds: {f.bounds}\nApply offsets: {l_bound_offset, u_bound_offset}')
        for i in range(f.dim):
            f._bounds[i] = (f._bounds[i][0] + l_bound_offset, f._bounds[i][1] + u_bound_offset)
            assert f._bounds[i][1] > f._bounds[i][0]
        f.register_buffer(
            "bounds", torch.tensor(f._bounds, dtype=torch.float).transpose(-1, -2)
        )
        print(f'New bounds: {f.bounds}')
        # Make sure that there is still at least one optimizer in the domain
        optimizers = []
        for optimizer in f._optimizers:
            in_domain = True
            for i in range(len(optimizer)):
                in_domain &= f._bounds[i][0] <= optimizer[i] <= f._bounds[i][1]
            if in_domain:
                optimizers.append(optimizer)
        if len(optimizers) == 0:
            raise ValueError('New bounds are such that no optimizers lay in the new domain.')
        f._optimizers = optimizers
        f.register_buffer(
            "optimizers", torch.tensor(f._optimizers, dtype=torch.float))
    return f


def query_covar(covar_name: str, X, Y, scale: bool, **kwargs) -> Kernel:
    """ Get covariance module

    Args:
        covar_name: name of the kernel to use ('matern-5/2', 'rbf')
        X: input points at which observations have been gathered
        Y: observations
        scale: whether to use a scaled GP

    Returns:
        An instance of GPyTorch kernel
    """
    ard_num_dims = X.shape[-1]
    aug_batch_shape = X.shape[:-2]
    num_outputs = Y.shape[-1]
    if num_outputs > 1:
        aug_batch_shape += torch.Size([num_outputs])
    lengthscale_prior = GammaPrior(3.0, 6.0)

    kws = dict(ard_num_dims=ard_num_dims, batch_shape=aug_batch_shape,
               lengthscale_prior=kwargs.pop('lengthscale_prior', lengthscale_prior))
    if covar_name.lower()[:6] == 'matern':
        kernel_class = MaternKernel
        if covar_name[-3:] == '5/2':
            kws['nu'] = 2.5
        elif covar_name[-3:] == '3/2':
            kws['nu'] = 1.5
        elif covar_name[-3:] == '1/2':
            kws['nu'] = .5
        else:
            raise ValueError(covar_name)
    elif covar_name.lower() == 'rbf':
        kernel_class = RBFKernel
    else:
        raise ValueError(covar_name)
    kws.update(**kwargs)
    outputscale_prior = kws.pop('outputscale_prior', GammaPrior(2.0, 0.15))

    base_kernel = kernel_class(**kws)
    if not scale:
        return base_kernel

    return ScaleKernel(base_kernel, batch_shape=aug_batch_shape, outputscale_prior=outputscale_prior)


def query_AcqFunc(acq_func: str, **acq_func_kwargs) -> Type[MCAcquisitionFunction]:
    """ Return the class of Acquisition function """
    if acq_func == "qMaxValueEntropy":
        acq_func_kwargs["num_candidates"] = acq_func_kwargs.get("candidate_set", 100)
    if hasattr(acquisition, acq_func):
        return getattr(acquisition, acq_func)
    elif hasattr(comp_acquisition, acq_func):
        return getattr(comp_acquisition, acq_func)
    else:
        raise ValueError(f'{acq_func} not found.')


def query_optimizer(optimizer: str) -> Type[Optimizer]:
    """ Get the class of Optimizer associated to `optimizer` """

    if hasattr(torch.optim, optimizer):
        return getattr(torch.optim, optimizer)
    elif hasattr(custom_optimizer, optimizer):
        return getattr(custom_optimizer, optimizer)
    if hasattr(evolution_opt, optimizer):
        return getattr(evolution_opt, optimizer)
    else:
        raise ValueError(f"Unavailable optimizer: {optimizer}")


def query_scheduler(scheduler: str) -> Optional[Type[lr_scheduler._LRScheduler]]:
    """ Get the class of optimizer scheduler associated to `scheduler` """
    if scheduler is None or scheduler == 'None':
        return None
    if hasattr(lr_scheduler, scheduler):
        return getattr(lr_scheduler, scheduler)
    else:
        raise ValueError(f"Unavailable optimizer: {scheduler}")
