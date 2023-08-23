# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import math
from typing import Optional, Union, List, Callable, Dict

import torch

from mcbo.acq_funcs import acq_factory
from mcbo.acq_optimizers.local_search_acq_optimizer import LsAcqOptimizer
from mcbo.models import ComboEnsembleGPModel
from mcbo.optimizers import BoBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager
from mcbo.utils.graph_utils import laplacian_eigen_decomposition


class COMBO(BoBase):

    @property
    def name(self) -> str:
        if self.use_tr:
            name = f'GP (Diffusion) - Tr-based LS acq optim'
        else:
            name = f'COMBO'
        return name

    def get_name(self, no_alias: bool = False) -> str:
        if no_alias and self.name == "COMBO":
            return f'GP (Diffusion) - LS acq optim'
        return self.name

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 n_init: int,
                 n_models: int = 10,
                 model_noise_lb: float = 1e-5,
                 model_n_burn: int = 0,
                 model_n_burn_init: int = 100,
                 model_max_training_dataset_size: int = 1000,
                 model_verbose: bool = False,
                 acq_name: str = 'ei',
                 acq_optim_n_random_vertices: int = 20000,
                 acq_optim_n_greedy_ascent_init: int = 20,
                 acq_optim_n_spray: int = 10,
                 acq_optim_max_n_ascent: float = float('inf'),
                 use_tr: bool = False,
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
        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'COMBO only supports nominal and ordinal variables.'

        if use_tr:

            assert search_space.num_ordinal == 0, 'The Casmopolitan trust region manager does not support ordinal variables'

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

        # Eigen decomposition of the graph laplacian
        n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
            search_space=search_space, device=device)

        model = ComboEnsembleGPModel(
            search_space=search_space,
            fourier_freq_list=fourier_freq_list,
            fourier_basis_list=fourier_basis_list,
            n_vertices=n_vertices,
            adjacency_mat_list=adjacency_mat_list,
            n_models=n_models,
            noise_lb=model_noise_lb,
            n_burn=model_n_burn,
            n_burn_init=model_n_burn_init,
            max_training_dataset_size=model_max_training_dataset_size,
            verbose=model_verbose,
            dtype=dtype,
            device=device,
        )

        # Initialise the acquisition function
        acq_func = acq_factory(acq_func_id=acq_name)

        acq_optim = LsAcqOptimizer(
            search_space=search_space,
            input_constraints=input_constraints,
            adjacency_mat_list=adjacency_mat_list,
            n_vertices=n_vertices,
            n_random_vertices=acq_optim_n_random_vertices,
            n_greedy_ascent_init=acq_optim_n_greedy_ascent_init,
            n_spray=acq_optim_n_spray,
            max_n_ascent=acq_optim_max_n_ascent,
            dtype=dtype
        )

        if use_tr:
            # Initialise the trust region manager
            tr_model = copy.deepcopy(model)

            tr_acq_func = acq_factory(acq_func_id=tr_restart_acq_name)

            tr_manager = CasmopolitanTrManager(search_space=search_space,
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
                                               device=device)
        else:
            tr_manager = None

        self.use_tr = use_tr

        super(COMBO, self).__init__(
            search_space=search_space,
            n_init=n_init,
            model=model,
            acq_func=acq_func,
            acq_optim=acq_optim,
            input_constraints=input_constraints,
            tr_manager=tr_manager,
            dtype=dtype,
            device=device
        )
