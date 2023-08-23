# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import time
from typing import Optional, Dict, Callable, List

import numpy as np
import pandas as pd
import torch

from mcbo.acq_funcs import AcqBase
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.optimizers.optimizer_base import OptimizerBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager
from mcbo.trust_region.tr_manager_base import TrManagerBase
from mcbo.utils.model_utils import move_model_to_device


class BoBase(OptimizerBase):

    @property
    def name(self) -> str:
        name = ""
        name += self.model.name
        name += " - "
        if self.tr_manager is not None and self.tr_name == 'basic':
            name += f"Tr-based "
        elif self.tr_manager is not None:
            name += self.tr_name + " "
        name += f"{self.acq_optimizer.name} acq optim"
        if self.init_sampling_strategy == "uniform":
            pass
        else:
            name += f" - sample {self.init_sampling_strategy}"
        return name

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def acq_opt_name(self) -> str:
        return self.acq_optimizer.name

    @property
    def tr_name(self) -> str:
        if self.tr_manager is None:
            return "no-tr"
        elif isinstance(self.tr_manager, CasmopolitanTrManager):
            return "basic"
        else:
            raise ValueError(self.tr_manager)

    @property
    def acq_func_name(self) -> str:
        return self.acq_func.name

    def get_linestyle_tr_based(self, non_tr_linestyle: str = "-", tr_linestyle: str = "--") -> str:
        if self.tr_manager is None:
            return non_tr_linestyle
        return tr_linestyle

    def get_color_acq_opt_based(self, color_dict: Optional[Dict[str, str]] = None) -> str:
        return self.acq_optimizer.get_color_1()

    def __init__(self,
                 search_space: SearchSpace,
                 n_init: int,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_optim: AcqOptimizerBase,
                 input_constraints: Optional[List[Callable[[Dict], bool]]] = None,
                 tr_manager: Optional[TrManagerBase] = None,
                 init_sampling_strategy: str = "uniform",
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu')
                 ):
        """
        Base Bayesian optimization optimizer

        Args:
            input_constraints: list of funcs taking a point as input and outputting whether the point
                                       is valid or not
            init_sampling_strategy: strategy to sample the first points to suggest (uniform, sobol, or sobol_scramble)

        """

        super(BoBase, self).__init__(
            search_space=search_space, dtype=dtype, input_constraints=input_constraints
        )

        assert isinstance(n_init, int) and n_init > 0
        assert isinstance(search_space, SearchSpace)
        assert isinstance(model, ModelBase)
        assert isinstance(acq_func, AcqBase)
        assert isinstance(acq_optim, AcqOptimizerBase)
        assert isinstance(tr_manager, TrManagerBase) or tr_manager is None
        assert isinstance(dtype, torch.dtype) and dtype in [torch.float32, torch.float64]
        assert isinstance(device, torch.device)

        self.device = device

        self._init_model = copy.deepcopy(model)
        self._init_acq_optimizer = copy.deepcopy(acq_optim)

        self.model = model
        self.acq_func = acq_func
        self.acq_optimizer = acq_optim
        self.tr_manager = tr_manager

        self.init_sampling_strategy = init_sampling_strategy
        self.n_init = n_init

        point_sampler = self.get_point_sampler()
        self.x_init = self.sample_input_valid_points(
            n_points=self.n_init,
            point_sampler=point_sampler
        )

        self.fit_time = []  # time taken to fit the surrogate model
        self.acq_time = []  # time taken to run acquisition function optimization
        self.observe_time = []  # time taken to observe a new value

    def get_point_sampler(self) -> Callable[[int], pd.DataFrame]:
        if self.init_sampling_strategy == "uniform":
            point_sampler = self.search_space.sample
        elif self.init_sampling_strategy == "sobol_scramble":
            soboleng = torch.quasirandom.SobolEngine(dimension=self.search_space.num_dims, scramble=True)
            point_sampler = lambda n_samples: self.search_space.inverse_transform(
                x=soboleng.draw(n_samples) * (
                        self.search_space.transfo_ub - self.search_space.transfo_lb) + self.search_space.transfo_lb
            )
        elif self.init_sampling_strategy == "sobol":
            soboleng = torch.quasirandom.SobolEngine(dimension=self.search_space.num_dims, scramble=False)
            point_sampler = lambda n_samples: self.search_space.inverse_transform(
                x=soboleng.draw(n_samples) * (
                        self.search_space.transfo_ub - self.search_space.transfo_lb) + self.search_space.transfo_lb
            )
        else:
            raise ValueError(self.init_sampling_strategy)
        return point_sampler

    def restart(self):
        self.observe_time = []
        self.acq_time = []
        self.fit_time = []
        self._restart()
        point_sampler = self.get_point_sampler()
        self.x_init = self.sample_input_valid_points(
            n_points=self.n_init,
            point_sampler=point_sampler
        )
        self.model = copy.deepcopy(self._init_model)
        self.acq_optimizer = copy.deepcopy(self._init_acq_optimizer)
        if self.tr_manager is not None:
            self.tr_manager.restart()

    def set_x_init(self, x: pd.DataFrame):
        assert x.ndim == 2
        assert x.shape[1] == self.search_space.num_dims
        self.x_init = x

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert y.ndim == 2
        assert x.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)
        self.x_init = self.x_init[len(x):]

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        self.data_buffer.append(x, y)

        if self.tr_manager is not None:
            self.tr_manager.append(x, y)

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:

        if self.tr_manager is not None:
            trigger_tr_reset = False
            for variable_type in self.tr_manager.variable_types:
                if self.tr_manager.radii[variable_type] < self.tr_manager.min_radii[variable_type]:
                    trigger_tr_reset = True
                    break

            if trigger_tr_reset:
                self.x_init = self.tr_manager.suggest_new_tr(
                    n_init=self.n_init,
                    observed_data_buffer=self.data_buffer,
                    input_constraints=self.input_constraints
                )

        # Create a Dataframe that will store the candidates
        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # Return as many points from initialisation as possible
        if len(self.x_init) and n_remaining:
            n = min(n_suggestions, len(self.x_init))
            x_next.iloc[idx: idx + n] = self.x_init.iloc[[i for i in range(0, n)]]
            self.x_init = self.x_init.drop(self.x_init.index[[i for i in range(0, n)]], inplace=False).reset_index(
                drop=True)
            idx += n
            n_remaining -= n

        # Sanity check
        if n_remaining and len(self.data_buffer) == 0:
            raise Exception('n_suggestion is larger than n_init and there is no data to fit a surrogate model to')

        # Get remaining points using standard BO loop
        if n_remaining:

            torch.cuda.empty_cache()  # Clear cached memory

            if self.tr_manager is not None:
                data_buffer = self.tr_manager.data_buffer
            else:
                data_buffer = self.data_buffer

            move_model_to_device(model=self.model, data_buffer=data_buffer, target_device=self.device)

            # Used to conduct and pre-fitting operations, such as creating a new model
            self.model.pre_fit_method(x=data_buffer.x, y=data_buffer.y)

            time_ref = time.time()
            # Fit the model
            _ = self.model.fit(x=data_buffer.x, y=data_buffer.y)
            self.fit_time.append(time.time() - time_ref)

            # Grab the current best x and y for acquisition evaluation and optimization
            best_x, best_y = self.get_best_x_and_y()
            acq_evaluate_kwargs = {'best_y': best_y}

            torch.cuda.empty_cache()  # Clear cached memory

            time_ref = time.time()
            # Optimise the acquisition function
            x_remaining = self.acq_optimizer.optimize(
                x=best_x, n_suggestions=n_remaining,
                x_observed=self.data_buffer.x,
                model=self.model,
                acq_func=self.acq_func,
                acq_evaluate_kwargs=acq_evaluate_kwargs,
                tr_manager=self.tr_manager
            )
            self.acq_time.append(time.time() - time_ref)

            x_next[idx: idx + n_remaining] = self.search_space.inverse_transform(x_remaining)

        return x_next

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:
        time_ref = time.time()

        is_valid = self.input_eval_from_origx(x=x)
        assert np.all(is_valid), is_valid

        # Transform x and y to torch tensors
        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data and to the trust region manager
        self.data_buffer.append(x, y)

        if self.tr_manager is not None:
            if len(self.tr_manager.data_buffer) > self.n_init:
                self.tr_manager.adjust_tr_radii(y)
            self.tr_manager.append(x, y)
            self.tr_manager.adjust_tr_center()

        # update best x and y
        if self.best_y is None:
            idx = y.flatten().argmin()
            self.best_y = y[idx, 0].item()
            self._best_x = x[idx: idx + 1]

        else:
            idx = y.flatten().argmin()
            y_ = y[idx, 0].item()

            if y_ < self.best_y:
                self.best_y = y_
                self._best_x = x[idx: idx + 1]

        # Used to update internal state of the optimizer if needed
        self.acq_optimizer.post_observe_method(x, y, self.data_buffer, self.n_init)

        self.observe_time.append(time.time() - time_ref)

    def get_best_x_and_y(self):
        """
        :return: Returns best x and best y used for acquisition optimization.
        """
        if self.tr_manager is None:
            x, y = self.data_buffer.x, self.data_buffer.y

        else:
            x, y = self.tr_manager.data_buffer.x, self.tr_manager.data_buffer.y

        idx = y.argmin()
        best_x = x[idx]
        best_y = y[idx]

        return best_x, best_y

    @property
    def is_numeric(self) -> bool:
        return self.search_space.num_numeric > 0

    @property
    def is_nominal(self) -> bool:
        return self.search_space.num_nominal > 0

    @property
    def is_mixed(self) -> bool:
        return self.is_nominal and self.is_numeric

    def get_time_dicts(self) -> Dict[str, List[float]]:
        return {
            "observation_time": self.observe_time,
            "acquisition_time": self.acq_time,
            "fit_time": self.fit_time,
        }

    def set_time_from_dict(self, time_dict: Dict[str, List[float]]) -> None:
        self.observe_time = time_dict["observation_time"]
        self.acq_time = time_dict["acquisition_time"]
        self.fit_time = time_dict["fit_time"]
