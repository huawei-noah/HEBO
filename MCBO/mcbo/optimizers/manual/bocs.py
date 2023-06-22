# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import math
import warnings
from typing import Optional, Union, List, Callable, Dict

import torch

from mcbo.acq_funcs import acq_factory
from mcbo.acq_optimizers.simulated_annealing_acq_optimizer import SimulatedAnnealingAcqOptimizer
from mcbo.models import LinRegModel
from mcbo.optimizers import BoBase
from mcbo.search_space import SearchSpace
from mcbo.search_space.params.bool_param import BoolPara
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager


class BOCS(BoBase):
    @property
    def name(self) -> str:
        if self.use_tr:
            name = f'LR ({self.model_estimator}) - Tr-Based SA acq optim'
        elif self.model_estimator == "sparse_horseshoe":  # Standard BOCS
            name = "BOCS"
        else:
            name = f'BOCS ({self.model_estimator})'
        return name

    def get_name(self, no_alias: bool = False) -> str:
        if no_alias:
            return f'LR ({self.model_estimator}) - {"Tr-Based " if self.use_tr else ""}SA acq optim'
        return self.name

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 n_init: int,
                 model_order: int = 2,
                 model_estimator: str = 'sparse_horseshoe',
                 model_a_prior: float = 2.,
                 model_b_prior: float = 1.,
                 model_sparse_horseshoe_threshold: float = 0.1,
                 model_n_gibbs: int = int(1e3),
                 acq_optim_num_iter: int = 100,
                 acq_optim_n_restarts: int = 3,
                 acq_optim_init_temp: int = 1,
                 acq_optim_tolerance: int = 100,
                 use_tr: bool = False,
                 tr_restart_n_cand: Optional[int] = None,
                 tr_min_nominal_radius: Optional[Union[int, float]] = None,
                 tr_max_nominal_radius: Optional[Union[int, float]] = None,
                 tr_init_nominal_radius: Optional[Union[int, float]] = None,
                 tr_radius_multiplier: Optional[float] = None,
                 tr_succ_tol: Optional[int] = None,
                 tr_fail_tol: Optional[int] = None,
                 tr_verbose: bool = False,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu')
                 ):

        assert search_space.num_nominal == search_space.num_params, 'BOCS only supports nominal variables.'

        # Check if the problem is purely binary
        binary_problem = True
        for param_name in search_space.params:
            binary_problem = binary_problem and isinstance(search_space.params[param_name], BoolPara)
            if not binary_problem:
                break

        if binary_problem:
            warning_message = 'This is the general form implementation of BOCS (see Appendix A of ' + \
                              'https://arxiv.org/abs/1806.08838), which differs from the standard implementation ' + \
                              'for purely binary problems. The differences are: (1) binary variables are ' + \
                              'represented by their one hot encoding, and (2) SA is used to optimize the acquisition' + \
                              'in place of SDP.'
            warnings.warn(warning_message, category=UserWarning)

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

        self.model_estimator = model_estimator
        self.use_tr = use_tr

        model = LinRegModel(
            search_space=search_space,
            order=model_order,
            estimator=model_estimator,
            a_prior=model_a_prior,
            b_prior=model_b_prior,
            sparse_horseshoe_threshold=model_sparse_horseshoe_threshold,
            n_gibbs=model_n_gibbs,
            dtype=dtype,
            device=device
        )

        acq_func = acq_factory('ts')

        acq_optim = SimulatedAnnealingAcqOptimizer(
            search_space=search_space,
            input_constraints=input_constraints,
            num_iter=acq_optim_num_iter,
            init_temp=acq_optim_init_temp,
            tolerance=acq_optim_tolerance,
            n_restarts=acq_optim_n_restarts,
            dtype=dtype
        )

        if use_tr:

            # Initialise the trust region manager
            tr_model = copy.deepcopy(model)

            tr_acq_func = acq_factory('ts')

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

        super(BOCS, self).__init__(
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
