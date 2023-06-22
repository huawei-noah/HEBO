# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
from typing import Union, Optional, List, Callable, Dict

import math
import torch
from gpytorch.constraints import Interval
from gpytorch.priors import Prior

from mcbo.acq_funcs.factory import acq_factory
from mcbo.acq_optimizers.interleaved_search_acq_optimizer import InterleavedSearchAcqOptimizer
from mcbo.models import ExactGPModel
from mcbo.models.gp.kernel_factory import mixture_kernel_factory
from mcbo.optimizers import BoBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager


class BOiLS(BoBase):

    @property
    def name(self) -> str:
        return self.get_name(no_alias=False)

    def get_name(self, no_alias: bool = False) -> str:
        if not no_alias:
            if self.use_tr:
                return "BOiLS"

        tr_prefix = "Tr-based " if self.use_tr else ""

        return f'GP (SSK) - {tr_prefix}IS acq optim'

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 n_init: int,
                 model_max_subsequence_length: int = 3,
                 model_kernel_gap_decay: float = 0.5,
                 model_kernel_match_decay: float = 0.8,
                 model_normalize_kernel_output: bool = True,
                 model_noise_prior: Optional[Prior] = None,
                 model_noise_constr: Optional[Interval] = None,
                 model_noise_lb: float = 1e-5,
                 model_pred_likelihood: bool = True,
                 model_optimizer: str = 'adam',
                 model_lr: float = 3e-2,
                 model_num_epochs: int = 100,
                 model_max_cholesky_size: int = 2000,
                 model_max_training_dataset_size: int = 1000,
                 model_max_batch_size: int = 50,
                 ls_acq_name: str = 'ei',
                 acq_optim_n_iter: int = 100,
                 acq_optim_n_restarts: int = 3,
                 acq_optim_max_n_perturb_num: int = 20,
                 acq_optim_num_optimizer: str = 'adam',
                 acq_optim_num_lr: Optional[float] = 1e-3,
                 acq_optim_nominal_tol: int = 100,
                 use_tr: bool = True,
                 tr_restart_acq_name: str = 'lcb',
                 tr_restart_n_cand: Optional[int] = None,
                 tr_min_nominal_radius: Optional[Union[int, float]] = None,
                 tr_max_nominal_radius: Optional[Union[int, float]] = None,
                 tr_init_nominal_radius: Optional[Union[int, float]] = None,
                 tr_radius_multiplier: Optional[float] = None,
                 tr_succ_tol: Optional[int] = None,
                 tr_fail_tol: Optional[int] = None,
                 tr_verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):

        assert search_space.num_nominal == search_space.num_params, \
            'BOiLS only supports nominal variables. To use BOiLS, define the search space such that ' \
            'the entire string' + \
            'is composed of only nominal variables. The alphabet for each position in the string is defined via the' + \
            ' \'categories\' argument of that parameter.'

        alphabet = search_space.params[search_space.param_names[0]].categories
        for param in search_space.params:
            assert search_space.params[param].categories == alphabet, \
                '\'categories\' must be the same for each of the nominal variables.'

        if use_tr:
            if tr_restart_n_cand is None:
                tr_restart_n_cand = min(100 * search_space.num_dims, 5000)
            else:
                assert isinstance(tr_restart_n_cand, int)
                assert tr_restart_n_cand > 0

            if search_space.num_nominal > 1:
                if tr_min_nominal_radius is None:
                    tr_min_nominal_radius = 1
                else:
                    assert 1 <= tr_min_nominal_radius <= search_space.num_nominal

                if tr_max_nominal_radius is None:
                    tr_max_nominal_radius = search_space.num_nominal
                else:
                    assert 1 <= tr_max_nominal_radius <= search_space.num_nominal

                if tr_init_nominal_radius is None:
                    tr_init_nominal_radius = math.ceil(0.8 * tr_max_nominal_radius)
                else:
                    assert tr_min_nominal_radius <= tr_init_nominal_radius <= tr_max_nominal_radius

                assert tr_min_nominal_radius < tr_init_nominal_radius <= tr_max_nominal_radius
            else:
                tr_min_nominal_radius = tr_init_nominal_radius = tr_max_nominal_radius = None

            if tr_radius_multiplier is None:
                tr_radius_multiplier = 1.5

            if tr_succ_tol is None:
                tr_succ_tol = 3

            if tr_fail_tol is None:
                tr_fail_tol = 40

        # Initialise the model
        alphabet_size = len(alphabet)

        nominal_kernel_kwargs = dict(
            seq_length=search_space.num_dims,
            alphabet_size=alphabet_size,
            gap_decay=model_kernel_gap_decay,
            match_decay=model_kernel_match_decay,
            max_subsequence_length=model_max_subsequence_length,
            normalize=model_normalize_kernel_output
        )

        kernel = mixture_kernel_factory(
            search_space=search_space,
            nominal_kernel_name="ssk",
            nominal_kernel_kwargs=nominal_kernel_kwargs
        )

        model = ExactGPModel(
            search_space=search_space,
            num_out=1,
            kernel=kernel,
            noise_prior=model_noise_prior,
            noise_constr=model_noise_constr,
            noise_lb=model_noise_lb,
            pred_likelihood=model_pred_likelihood,
            lr=model_lr,
            num_epochs=model_num_epochs,
            optimizer=model_optimizer,
            max_cholesky_size=model_max_cholesky_size,
            max_training_dataset_size=model_max_training_dataset_size,
            max_batch_size=model_max_batch_size,
            dtype=dtype,
            device=device
        )

        # Initialise the acquisition function
        acq_func = acq_factory(acq_func_id=ls_acq_name)

        # Initialise the acquisition optimizer
        acq_optim = InterleavedSearchAcqOptimizer(
            search_space=search_space,
            input_constraints=input_constraints,
            n_iter=acq_optim_n_iter,
            n_restarts=acq_optim_n_restarts,
            max_n_perturb_num=acq_optim_max_n_perturb_num,
            num_optimizer=acq_optim_num_optimizer,
            num_lr=acq_optim_num_lr,
            nominal_tol=acq_optim_nominal_tol,
            dtype=dtype
        )

        if use_tr:
            # Initialise the trust region manager
            tr_model = copy.deepcopy(model)

            tr_acq_func = acq_factory(acq_func_id=tr_restart_acq_name)

            tr_manager = CasmopolitanTrManager(
                search_space=search_space,
                model=tr_model,
                acq_func=tr_acq_func,
                n_init=n_init,
                min_num_radius=0,  # predefined as not relevant (no numerical variables)
                max_num_radius=1,  # predefined as not relevant (no numerical variables)
                init_num_radius=0.8,  # predefined as not relevant (no numerical variables)
                min_nominal_radius=tr_min_nominal_radius,
                max_nominal_radius=tr_max_nominal_radius,
                init_nominal_radius=tr_init_nominal_radius,
                radius_multiplier=tr_radius_multiplier,
                succ_tol=tr_succ_tol,
                fail_tol=tr_fail_tol,
                restart_n_cand=tr_restart_n_cand,
                verbose=tr_verbose,
                dtype=dtype,
                device=device
            )
        else:
            tr_manager = None

        self.use_tr = use_tr

        super(BOiLS, self).__init__(
            search_space=search_space,
            input_constraints=input_constraints,
            n_init=n_init,
            model=model,
            acq_func=acq_func,
            acq_optim=acq_optim,
            tr_manager=tr_manager,
            dtype=dtype,
            device=device
        )
