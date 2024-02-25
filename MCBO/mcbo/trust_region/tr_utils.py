# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional, List, Union, Tuple

import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from torch.quasirandom import SobolEngine

from mcbo.models import ExactGPModel, ModelBase
from mcbo.models.gp.kernels import MixtureKernel, ConditionalTransformedOverlapKernel, DecompositionKernel
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.utils.discrete_vars_utils import round_discrete_vars


def get_num_tr_bounds(
        x_num: torch.Tensor,
        tr_manager: TrManagerBase,
        is_numeric: bool,
        is_mixed: bool,
        kernel: Optional[MixtureKernel] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 'numeric' in tr_manager.radii, "numeric not in radii"
    # This function requires the mixture kernel or the RBF or Matérn kernel to use lengthscales as weights
    if is_numeric:
        weights = get_numdim_weights(
            num_dim=x_num.shape[-1], is_numeric=is_numeric, is_mixed=is_mixed, kernel=kernel
        )

        # continuous variables
        lb = torch.clip(x_num - weights.to(x_num) * tr_manager.radii['numeric'] / 2.0, 0.0, 1.0)
        ub = torch.clip(x_num + weights.to(x_num) * tr_manager.radii['numeric'] / 2.0, 0.0, 1.0)

    else:
        lb = torch.zeros_like(x_num)
        ub = torch.ones_like(x_num)

    return lb, ub


def get_numdim_weights(num_dim: int, is_numeric: bool, is_mixed: bool, kernel: Optional[MixtureKernel]) -> torch.Tensor:
    if kernel is not None:
        if isinstance(kernel, ScaleKernel):
            kernel = kernel.base_kernel
        valid_kernel = False
        if is_mixed:
            if isinstance(kernel, MixtureKernel):
                valid_kernel = True
            elif isinstance(kernel, (ConditionalTransformedOverlapKernel, DecompositionKernel)):
                valid_kernel = True
            else:
                raise ValueError(kernel)
        elif is_numeric:
            if isinstance(kernel, (RBFKernel, MaternKernel, DecompositionKernel)):
                valid_kernel = True
            else:
                raise ValueError(kernel)
        if not valid_kernel:
            kernel = None

    if is_numeric:

        if kernel is not None:

            if is_mixed:
                if isinstance(kernel, (ConditionalTransformedOverlapKernel, DecompositionKernel)):
                    weights = kernel.get_lengthcales_numerical_dims().detach()
                elif isinstance(kernel, MixtureKernel):
                    weights = kernel.numeric_kernel.base_kernel.lengthscale.detach().cpu()
                else:
                    raise ValueError(kernel)

            else:
                if hasattr(kernel, "lengthscale"):
                    weights = kernel.lengthscale.detach().cpu()
                else:
                    raise ValueError(kernel)

            # Normalise the weights so that we have weights.prod() = 1
            weights = weights[0] / weights.mean()
            weights = weights / torch.pow(torch.prod(weights), 1 / len(weights))

        else:
            weights = torch.ones(num_dim)

    return weights


def sample_numeric_and_nominal_within_tr(
        x_centre: torch.Tensor,
        search_space: SearchSpace,
        tr_manager: TrManagerBase,
        n_points: int,
        numeric_dims: List[int],
        discrete_choices: Union[torch.FloatTensor, List[torch.Tensor]],
        seq_dims: Optional[Union[np.ndarray, List[int]]] = None,
        max_n_perturb_num: int = 20,
        model: Optional[ModelBase] = None,
        return_numeric_bounds: bool = False,
):
    is_numeric = search_space.num_numeric > 0
    is_mixed = is_numeric and search_space.num_nominal > 0
    x_centre = x_centre.clone() * torch.ones((n_points, search_space.num_dims)).to(x_centre)

    if search_space.num_numeric > 0:

        if (model is not None) and isinstance(model, ExactGPModel):
            kernel = model.gp.kernel
        else:
            kernel = None

        num_lb, num_ub = get_num_tr_bounds(x_centre[0, numeric_dims], tr_manager, is_numeric, is_mixed, kernel)

        seed = np.random.randint(int(1e6))
        sobol_engine = SobolEngine(search_space.num_numeric, scramble=True, seed=seed)
        pert = sobol_engine.draw(n_points).to(x_centre)
        pert = (num_ub - num_lb) * pert + num_lb

        perturb_prob = min(max_n_perturb_num / search_space.num_numeric, 1.0)

        mask = torch.rand(n_points, search_space.num_numeric) <= perturb_prob
        ind = torch.where(torch.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, search_space.num_numeric, size=len(ind))] = 1

        x_centre[:, numeric_dims] = torch.where(mask.to(x_centre.device), pert, x_centre[:, numeric_dims])
        x_centre = round_discrete_vars(x_centre, search_space.disc_dims, discrete_choices)

    if seq_dims is None:
        seq_dims = []
    nominal_non_seq_dims = [nominal_dim for nominal_dim in search_space.nominal_dims if nominal_dim not in seq_dims]
    if len(nominal_non_seq_dims) > 0:
        low = 0 if search_space.num_numeric > 0 else 1

        n_perturb_nominal = np.random.randint(low=low, high=tr_manager.get_nominal_radius() + 1, size=n_points)

        for i in range(n_points):
            for j in range(n_perturb_nominal[i]):
                dim = np.random.choice(nominal_non_seq_dims, replace=True)
                param = search_space.params[search_space.param_names[dim]]
                choices = [val for val in range(param.ub - param.lb + 1) if val != x_centre[i, dim].item()]
                x_centre[i, dim] = np.random.choice(choices, replace=True)

    if (len(seq_dims)) > 0:  # true for EDA task
        n_perturb_nominal = np.random.randint(low=0, high=tr_manager.radii['sequence'] + 1, size=n_points)
        from mcbo.search_space.search_space_eda import SearchSpaceEDA
        assert isinstance(search_space, SearchSpaceEDA)
        for i in range(n_points):
            for j in range(n_perturb_nominal[i]):
                dim = np.random.choice(seq_dims, replace=True)
                choices = [val.item() for val in
                           search_space.get_transformed_mutation_cand_values(transformed_x=x_centre[i], dim=dim) if
                           val != x_centre[i, dim].item()]
                if len(choices) > 0:
                    x_centre[i, dim] = np.random.choice(choices, replace=True)

    if return_numeric_bounds:
        if search_space.num_numeric > 0:
            return x_centre, num_lb, num_ub
        else:
            return x_centre, None, None
    else:
        return x_centre
