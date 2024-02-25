# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Optional, List, Callable, Dict, Tuple, Union

import numpy as np
import torch

from mcbo.acq_funcs import AcqBase
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.optimizers.non_bo.simulated_annealing import SimulatedAnnealing
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.data_buffer import DataBuffer
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class SimulatedAnnealingAcqOptimizer(AcqOptimizerBase):
    color_1: str = get_color(ind=4, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return SimulatedAnnealingAcqOptimizer.color_1

    @property
    def name(self) -> str:
        return f"SA"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 num_iter: int = 100,
                 init_temp: float = 1.,
                 tolerance: int = 100,
                 n_restarts: int = 3,
                 dtype: torch.dtype = torch.float64,
                 ):
        """
        Args:
            n_restarts: during acq function optimization, run SA from `n_restarts` starting points
        """
        self.num_iter = num_iter
        self.init_temp = init_temp
        self.tolerance = tolerance
        self.n_restarts = n_restarts

        self.numeric_dims = search_space.cont_dims + search_space.disc_dims
        self.cat_dims = np.sort(search_space.nominal_dims + search_space.ordinal_dims).tolist()
        self.discrete_choices = get_discrete_choices(search_space)

        super(SimulatedAnnealingAcqOptimizer, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )

        assert search_space.num_cont + search_space.num_disc + search_space.num_nominal == search_space.num_dims, \
            'Simulated Annealing is currently implemented for nominal, integer and continuous variables only'

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:

        # sample valid starting points
        if tr_manager:
            point_sampler = lambda num_points: self.search_space.inverse_transform(
                sample_numeric_and_nominal_within_tr(
                    x_centre=tr_manager.center,
                    search_space=self.search_space,
                    tr_manager=tr_manager,
                    n_points=num_points,
                    numeric_dims=self.numeric_dims,
                    discrete_choices=self.discrete_choices,
                    max_n_perturb_num=20,
                    model=model,
                    return_numeric_bounds=False
                )
            )
        else:
            point_sampler = self.search_space.sample
        start_points = self.search_space.transform(
            self.sample_input_valid_points(n_points=self.n_restarts, point_sampler=point_sampler)
        )

        start_points[0] = x
        best_acq_val = np.inf  # we minimize acquisition functions
        best_x = None
        for start_point in start_points:
            new_x, new_acq_val = self._optimize(
                x=start_point,
                n_suggestions=n_suggestions,
                x_observed=x_observed,
                model=model,
                acq_func=acq_func,
                acq_evaluate_kwargs=acq_evaluate_kwargs,
                tr_manager=tr_manager,
                **kwargs
            )

            if new_acq_val < best_acq_val:
                best_x = new_x
                best_acq_val = new_acq_val

        return best_x

    def _optimize(self, x: torch.Tensor, n_suggestions: int, x_observed: torch.Tensor, model: ModelBase,
                  acq_func: AcqBase, acq_evaluate_kwargs: dict,
                  tr_manager: Optional[TrManagerBase], **kwargs) -> Tuple[torch.Tensor, float]:
        """
        Optimize acquisition function from starting point `x`

        Returns:
            best_x: optimizer of acquisition function found by running SA
            best_v: associated acquistion function value
        """

        assert n_suggestions == 1, 'SA acquisition optimizer does not support suggesting batches of data'

        dtype = model.dtype

        sa = SimulatedAnnealing(
            search_space=self.search_space,
            input_constraints=self.input_constraints,
            obj_dims=self.obj_dims,
            out_upper_constr_vals=self.out_upper_constr_vals,
            out_constr_dims=self.out_constr_dims,
            fixed_tr_manager=tr_manager,
            init_temp=self.init_temp,
            tolerance=self.tolerance,
            store_observations=True,
            allow_repeating_suggestions=False,
            dtype=dtype
        )

        sa.x_init.iloc[0:1] = self.search_space.inverse_transform(x.unsqueeze(0))

        with torch.no_grad():
            for _ in range(self.num_iter):
                x_next = sa.suggest(1)
                y_next = acq_func(
                    x=self.search_space.transform(x_next).to(dtype),
                    model=model,
                    **acq_evaluate_kwargs
                ).view(-1, 1).detach().cpu().numpy()
                sa.observe(x_next, y_next)

        # Check if any of the samples was previous unobserved
        valid = False
        x_sa, y_sa = sa.data_buffer.x, sa.data_buffer.y
        indices = y_sa.flatten().argsort()
        for idx in indices:
            x = x_sa[idx]
            if torch.logical_not((x.unsqueeze(0) == x_observed).all(axis=1)).all():
                valid = True
                break

        # If a valid sample was still not found, suggest a random sample
        if not valid:
            if tr_manager:
                point_sampler = lambda num_points: self.search_space.inverse_transform(
                    sample_numeric_and_nominal_within_tr(
                        x_centre=tr_manager.center,
                        search_space=self.search_space,
                        tr_manager=tr_manager,
                        n_points=num_points,
                        numeric_dims=self.numeric_dims,
                        discrete_choices=self.discrete_choices,
                        max_n_perturb_num=20,
                        model=model,
                        return_numeric_bounds=False
                    )
                )
            else:
                point_sampler = self.search_space.sample
            x = self.search_space.transform(
                self.sample_input_valid_points(n_points=1, point_sampler=point_sampler)
            )
        else:
            x = x.unsqueeze(0)

        with torch.no_grad():
            acq_val = acq_func(x=x, model=model, **acq_evaluate_kwargs).item()
        return x, acq_val

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        """
        This function is used to set the initial temperature parameter of simulated annealing based on the observed
        data.

        :param x:
        :param y:
        :param data_buffer:
        :param n_init:
        :param kwargs:
        :return:
        """
        if len(data_buffer) == 1:
            self.init_temp = data_buffer.y[0].item()
        else:
            y = data_buffer.y
            init_temp = (y.max() - y.min()).item()
            init_temp = init_temp if init_temp != 0 else 1.
            self.init_temp = init_temp
