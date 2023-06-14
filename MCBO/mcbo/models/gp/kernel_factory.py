# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional, Dict, Any, Union

import numpy as np
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel

from mcbo.models.gp.kernels import DiffusionKernel, MixtureKernel, Overlap, TransformedOverlap, \
    SubStringKernel, HEDKernel
from mcbo.search_space import SearchSpace


def kernel_factory(
        kernel_name: str,
        active_dims: Optional[Union[list, np.ndarray]] = None,
        use_ard: bool = True,
        lengthscale_constraint: Optional[Interval] = None,
        outputscale_constraint: Optional[Interval] = None,
        **kwargs
) -> Optional[Kernel]:
    if active_dims is not None:
        if len(active_dims) == 0:
            return None

    ard_num_dims = len(active_dims) if use_ard else None

    # Kernels for numeric variables
    if kernel_name is None:
        kernel = None

    elif kernel_name == 'diffusion':
        assert 'fourier_freq_list' in kwargs
        assert 'fourier_basis_list' in kwargs
        kernel = DiffusionKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                                 fourier_freq_list=kwargs.get('fourier_freq_list'),
                                 fourier_basis_list=kwargs.get('fourier_basis_list'))

    elif kernel_name == 'rbf':
        kernel = RBFKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                           lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'mat52':
        kernel = MaternKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                              lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'overlap':
        kernel = Overlap(active_dims=active_dims, ard_num_dims=ard_num_dims,
                         lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'transformed_overlap':
        kernel = TransformedOverlap(active_dims=active_dims, ard_num_dims=ard_num_dims,
                                    lengthscale_constraint=lengthscale_constraint)

    elif kernel_name == 'ssk':
        assert 'search_space' in kwargs

        # Firstly check that the ssk kernel is applied to the nominal dimensions
        assert active_dims == kwargs.get(
            'search_space').nominal_dims, 'The SSK kernel can only be applied to nominal variables'

        # Secondly check that all of the ordinal dims share the same alphabet size
        alphabet_per_var = [kwargs.get('search_space').params[param_name].categories for param_name in
                            kwargs.get('search_space').nominal_names]
        assert all(alphabet == alphabet_per_var[0] for alphabet in
                   alphabet_per_var), 'The alphabet must be the same for each of the nominal variables'

        kernel = SubStringKernel(
            seq_length=len(active_dims),
            alphabet_size=len(alphabet_per_var[0]),
            gap_decay=0.5,
            match_decay=0.8,
            max_subsequence_length=len(active_dims) if len(active_dims) < 3 else 3,
            normalize=False,
            active_dims=active_dims
        )

    elif kernel_name == 'hed':
        assert active_dims == kwargs.get(
            'search_space').nominal_dims, f'The HED kernel can only be applied to nominal variables\n ' \
                                          f'Got\n\t{active_dims}\n and\n\t' \
                                          f'{kwargs.get("search_space").nominal_dims}'

        # Secondly check that all of the ordinal dims share the same alphabet size
        n_cats_per_dim = [len(kwargs.get('search_space').params[param_name].categories) for param_name in
                          kwargs.get('search_space').nominal_names]

        base_kernel = kwargs.get('hed_base_kernel')
        hed_num_embedders = kwargs.get('hed_num_embedders', 128)
        kernel = HEDKernel(
            base_kernel=base_kernel,
            hed_num_embedders=hed_num_embedders,
            n_cats_per_dim=n_cats_per_dim,
            active_dims=active_dims
        )

    else:
        raise NotImplementedError(f'{kernel_name} was not implemented')

    if kernel is not None:
        kernel = ScaleKernel(kernel, outputscale_constraint=outputscale_constraint)

    return kernel


def mixture_kernel_factory(
        search_space: SearchSpace,
        numeric_kernel_name: Optional[str] = None,
        numeric_kernel_use_ard: Optional[bool] = True,
        numeric_lengthscale_constraint: Optional[Interval] = None,
        nominal_kernel_name: Optional[str] = None,
        nominal_kernel_use_ard: Optional[bool] = True,
        nominal_lengthscale_constraint: Optional[Interval] = None,
        nominal_kernel_kwargs: Optional[Dict[str, Any]] = None,
        numeric_kernel_kwargs: Optional[Dict[str, Any]] = None
) -> Kernel:
    is_numeric = search_space.num_numeric > 0
    is_nominal = search_space.num_nominal > 0
    is_mixed = is_numeric and is_nominal

    if nominal_kernel_kwargs is None:
        nominal_kernel_kwargs = {}
    if numeric_kernel_kwargs is None:
        numeric_kernel_kwargs = {}

    if is_mixed:

        assert numeric_kernel_name is not None
        assert nominal_kernel_name is not None
        assert numeric_kernel_name in ['mat52', 'rbf'], numeric_kernel_name
        assert nominal_kernel_name in ['overlap', 'transformed_overlap', 'hed'], nominal_kernel_name

        num_dims = search_space.cont_dims + search_space.disc_dims
        nominal_dims = search_space.nominal_dims

        nominal_kernel = kernel_factory(
            kernel_name=nominal_kernel_name,
            active_dims=nominal_dims,
            use_ard=nominal_kernel_use_ard,
            lengthscale_constraint=nominal_lengthscale_constraint,
            outputscale_constraint=None,
            search_space=search_space,
            **nominal_kernel_kwargs
        )

        numeric_kernel = kernel_factory(
            kernel_name=numeric_kernel_name,
            active_dims=num_dims,
            use_ard=numeric_kernel_use_ard,
            lengthscale_constraint=numeric_lengthscale_constraint,
            outputscale_constraint=None,
            search_space=search_space,
            **numeric_kernel_kwargs
        )

        kernel = ScaleKernel(
            MixtureKernel(
                search_space=search_space,
                numeric_kernel=numeric_kernel,
                categorical_kernel=nominal_kernel,
            )
        )
    else:
        if is_numeric:

            assert numeric_kernel_name is not None
            active_dims = search_space.cont_dims + search_space.disc_dims

            kernel = kernel_factory(
                kernel_name=numeric_kernel_name,
                active_dims=active_dims,
                use_ard=numeric_kernel_use_ard,
                lengthscale_constraint=numeric_lengthscale_constraint,
                outputscale_constraint=None,
                search_space=search_space,
                **numeric_kernel_kwargs
            )

        elif is_nominal:

            assert nominal_kernel_name is not None
            kernel = kernel_factory(
                kernel_name=nominal_kernel_name, active_dims=search_space.nominal_dims,
                use_ard=nominal_kernel_use_ard,
                lengthscale_constraint=nominal_lengthscale_constraint,
                outputscale_constraint=None,
                search_space=search_space,
                **nominal_kernel_kwargs
            )

        else:
            raise ValueError("Not numeric nor nominal")

    return kernel
