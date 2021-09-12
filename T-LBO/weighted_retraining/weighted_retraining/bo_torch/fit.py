from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
from botorch.optim.numpy_converter import (
    TorchAttr,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
)
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [np.ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, np.ndarray]
]
TModToArray = Callable[
    [Module, Optional[ParameterBounds], Optional[Set[str]]],
    Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]],
]
TArrayToMod = Callable[[Module, np.ndarray, Dict[str, TorchAttr]], Module]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_gpytorch_torch(
        mll: MarginalLogLikelihood,
        bounds: Optional[ParameterBounds] = None,
        optimizer_cls: Optimizer = Adam,
        options: Optional[Dict[str, Any]] = None,
        track_iterations: bool = True,
        approx_mll: bool = True,
        clip_by_value: Optional[bool] = False,
        clip_by_norm: Optional[bool] = False,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        clip_norm: max norm param when clipping gradient of the kernel parameters w.r.t. mll
        clip_value: max value when clipping gradient of the kernel parameters w.r.t. mll
        clip_by_norm: whether to clip gradient of the kernel parameters w.r.t. mll by norm
        clip_by_value: whether to clip gradient of the kernel parameters w.r.t. mll by value
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())
    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not stop:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            loss.backward()
        if optim_options["disp"] and (
                (i + 1) % 10 == 0 or i == (optim_options["maxiter"] - 1)
        ):
            print(f"Iter {i + 1}/{optim_options['maxiter']}: {loss.item()}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))

        # for n, p in mll.named_parameters():
        #     if torch.isnan(p.grad).sum() > 0 or torch.isinf(p.grad).sum() > 0:  # or (p.grad.abs() < 1e-14).sum() > 0:
        #         print('================ Before Clipping ================')
        #         print(n)
        #         print(p.grad)

        if clip_by_value:
            assert not clip_by_norm, "Cannot clip both by value and norm."
            assert clip_value is not None, "clip_value must be provided."
            torch.nn.utils.clip_grad_value_(mll.parameters(), clip_value=clip_value)
        if clip_by_norm:
            assert not clip_by_value, "Cannot clip both by value and norm."
            assert clip_norm is not None, "clip_norm must be provided."
            torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=clip_norm)

        for n, p in mll.named_parameters():
            if torch.isnan(p.grad).sum() > 0 or torch.isinf(p.grad).sum() > 0:  # or (p.grad.abs() < 1e-14).sum() > 0:
                # print('================ After Clipping ================')
                # print(n)
                # print(p.grad)
                p.grad[torch.isnan(p.grad)] = 0.

        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    return mll, info_dict
