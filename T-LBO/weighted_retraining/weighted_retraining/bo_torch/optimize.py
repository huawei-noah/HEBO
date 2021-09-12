r"""
Methods for optimizing acquisition functions.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any, Iterable

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs, columnwise_clamp, fix_features
from torch import Tensor
from torch.optim import Optimizer


def optimize_acqf_torch(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
        batch_initial_conditions: Optional[Tensor] = None,
        return_best_only: bool = True,
        sequential: bool = False,
        verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        options: Options for candidate generation.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        verbose: verbosity option

        Returns:
            A two-element tuple containing

           - a `(num_restarts) x q x d`-dim tensor of generated candidates.
           - a tensor of associated acquisiton values. If `sequential=False`,
             this is a `(num_restarts)`-dim tensor of joint acquisition values
             (with explicit restart dimension if `return_best_only=False`). If
             `sequential=True`, this is a `q`-dim tensor of expected acquisition
             values conditional on having observed canidates `0,1,...,i-1`.
    """
    print(options)
    if sequential:
        if not return_best_only:
            raise NotImplementedError(
                "return_best_only=False only supported for joint optimization"
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        candidates = torch.tensor([])
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf_torch(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
                verbose=verbose,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i + 1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    if isinstance(acq_function, qKnowledgeGradient):
        ic_gen = gen_one_shot_kg_initial_conditions
    elif isinstance(acq_function, AcquisitionFunction):
        ic_gen = gen_batch_initial_conditions
    else:
        raise ValueError(acq_function)
    if batch_initial_conditions is None:
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )

    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    start_idcs = list(range(0, num_restarts, batch_limit))
    for start_idx in start_idcs:
        end_idx = min(start_idx + batch_limit, num_restarts)
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_torch(
            initial_conditions=batch_initial_conditions[start_idx:end_idx],
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("batch_limit", "nonnegative")
            },
            fixed_features=fixed_features,
            verbose=verbose,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {start_idx + 1} of {len(start_idcs)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def gen_candidates_torch(
        initial_conditions: Tensor,
        acquisition_function: Callable,
        lower_bounds: Optional[Union[float, Tensor]] = None,
        upper_bounds: Optional[Union[float, Tensor]] = None,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        options: Optional[Dict[str, Union[float, str]]] = None,
        verbose: bool = True,
        fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Iterable[Any]:  # -> Tuple[Tensor, Any, Optional[Tensor]]:
    r"""Generate a set of candidates using a `torch.optim` optimizer.

    Optimizes an acquisition function starting from a set of initial candidates
    using an optimizer from `torch.optim`.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        verbose: If True, provide verbose output.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.
    """
    options = options or {}
    _jitter = options.get('jitter', 0.)
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    ).requires_grad_(True)
    candidates = fix_features(clamped_candidates, fixed_features)

    bayes_optimizer = optimizer(
        params=[clamped_candidates], lr=options.get("lr", 0.025)
    )
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **options)
    )
    while not stop:
        i += 1
        batch_loss = acquisition_function(candidates)
        loss = -batch_loss.sum()

        if verbose:
            print("Iter: {} - Value: {:.3f}".format(i, -(loss.item())))

        if torch.isnan(loss):
            print('loss is nan, exiting optimization of the acquisition function.')
            break

        bayes_optimizer.zero_grad()
        loss.backward()
        if options.get('clip_gradient', False):
            torch.nn.utils.clip_grad_value_(clamped_candidates, clip_value=options.get('clip_value', 10.))
        bayes_optimizer.step()
        clamped_candidates.data = columnwise_clamp(
            clamped_candidates, lower_bounds + _jitter, upper_bounds - _jitter
        )
        candidates = fix_features(clamped_candidates, fixed_features)
        stop = stopping_criterion.evaluate(fvals=loss.detach())

    # clamped_candidates = columnwise_clamp(
    #     X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    # )

    with torch.no_grad():
        batch_acquisition = acquisition_function(candidates)

    return candidates, batch_acquisition
