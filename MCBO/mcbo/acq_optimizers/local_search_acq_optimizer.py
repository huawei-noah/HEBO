# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import os
import warnings
from typing import List, Optional, Callable, Dict, Union

import numpy as np
import torch

from mcbo.acq_funcs.acq_base import AcqBase
from mcbo.acq_optimizers.acq_optimizer_base import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.distance_metrics import hamming_distance
from mcbo.utils.graph_utils import cartesian_neighbors, cartesian_neighbors_center_attracted
from mcbo.utils.model_utils import add_hallucinations_and_retrain_model
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class LsAcqOptimizer(AcqOptimizerBase):
    color_1: str = get_color(ind=2, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return LsAcqOptimizer.color_1

    @property
    def name(self) -> str:
        return f"LS"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 adjacency_mat_list: List[torch.FloatTensor],
                 n_vertices: np.array,
                 n_random_vertices: int = 20000,
                 n_greedy_ascent_init: int = 20,
                 n_spray: int = 10,
                 max_n_ascent: float = float('inf'),
                 max_n_perturb_num: int = 20,
                 dtype: torch.dtype = torch.float64,
                 ):

        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
            'The greedy descent acquisition optimizer only supports nominal and ordinal variables.'

        super(LsAcqOptimizer, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )

        self.is_numeric = True if search_space.num_cont > 0 or search_space.num_disc > 0 else False
        self.is_nominal = True if search_space.num_nominal > 0 else False
        self.is_mixed = True if self.is_numeric and self.is_nominal else False
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.discrete_choices = get_discrete_choices(search_space)
        self.max_n_perturb_num = max_n_perturb_num

        self.n_spray = n_spray
        self.n_random_vertices = n_random_vertices
        self.n_greedy_ascent_init = n_greedy_ascent_init
        self.max_n_descent = max_n_ascent

        if self.n_greedy_ascent_init % 2 == 1:
            self.n_greedy_ascent_init += 1

        self.n_vertices = n_vertices
        self.adjacency_mat_list = adjacency_mat_list

        self.n_cpu = os.cpu_count()
        self.n_available_cores = min(10, self.n_cpu)

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

        if n_suggestions == 1:
            return self._optimize(
                x=x,
                n_suggestions=1,
                x_observed=x_observed,
                model=model,
                acq_func=acq_func,
                acq_evaluate_kwargs=acq_evaluate_kwargs,
                tr_manager=tr_manager
            )
        else:
            x_next = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
            model = copy.deepcopy(model)  # create a local copy of the model to be able to retrain it
            x_observed = x_observed.clone()

            for i in range(n_suggestions):
                x_ = self._optimize(
                    x=x,
                    n_suggestions=1,
                    x_observed=x_observed,
                    model=model,
                    acq_func=acq_func,
                    acq_evaluate_kwargs=acq_evaluate_kwargs,
                    tr_manager=tr_manager
                )
                x_next = torch.cat((x_next, x_), dim=0)

                # No need to add hallucinations during last iteration as the model will not be used
                if i < n_suggestions - 1:
                    x_observed = torch.cat((x_observed, x_), dim=0)
                    add_hallucinations_and_retrain_model(model, x_[0])

            return x_next

    def _optimize(self,
                  x: torch.Tensor,
                  n_suggestions: int,
                  x_observed: torch.Tensor,
                  model: ModelBase,
                  acq_func: AcqBase,
                  acq_evaluate_kwargs: dict,
                  tr_manager: Optional[TrManagerBase],
                  **kwargs
                  ) -> torch.Tensor:

        assert n_suggestions == 1, 'Greedy Ascent acquisition optimization does not support n_suggestions > 1'

        device, dtype = model.device, model.dtype
        tkwargs = dict(device=device, dtype=dtype)

        # Sample initial points
        if tr_manager:
            assert tr_manager.get_nominal_radius() > 0, "Cannot suggest any neighbors, TR should have been restarted"
            x_centre = x.clone()
            point_sampler = lambda n_points: self.search_space.inverse_transform(
                sample_numeric_and_nominal_within_tr(
                    x_centre=x_centre,
                    search_space=self.search_space,
                    tr_manager=tr_manager,
                    n_points=n_points,
                    numeric_dims=self.numeric_dims,
                    discrete_choices=self.discrete_choices,
                    max_n_perturb_num=self.max_n_perturb_num,
                    model=model,
                    return_numeric_bounds=False
                )
            )
        else:
            point_sampler = self.search_space.sample

        x_random = self.search_space.transform(
            self.sample_input_valid_points(
                n_points=self.n_random_vertices,
                point_sampler=point_sampler
            )
        ).to(**tkwargs)

        x_neighbours = cartesian_neighbors(x.long(), self.adjacency_mat_list).to(**tkwargs)
        valid_filter = np.all(self.input_eval_from_transfx(x_neighbours), axis=1)
        x_neighbours = x_neighbours[valid_filter]

        x_next = None
        if len(x_neighbours) == 0:
            warnings.warn("Could not find any valid neighbours for acquisition function acquisition.")
            indices = []
        else:
            shuffled_ind = list(range(x_neighbours.size(0)))
            np.random.shuffle(shuffled_ind)
            x_init_candidates = torch.cat(tuple([x_neighbours[shuffled_ind[:self.n_spray]], x_random]), dim=0)
            with torch.no_grad():
                acq_values = acq_func(x=x_init_candidates, model=model, **acq_evaluate_kwargs)

            non_nan_ind = ~torch.isnan(acq_values)
            x_init_candidates = x_init_candidates[non_nan_ind]
            acq_values = acq_values[non_nan_ind]

            acq_sorted, acq_sort_ind = torch.sort(acq_values, descending=False)
            x_init_candidates = x_init_candidates[acq_sort_ind]

            x_inits, acq_inits = x_init_candidates[:self.n_greedy_ascent_init], acq_sorted[:self.n_greedy_ascent_init]

            # Greedy Descent
            exhaustive_ls_return_values = [
                self._exhaustive_ls(
                    x_init=x_inits[i],
                    acq_func=acq_func,
                    model=model,
                    acq_evaluate_kwargs=acq_evaluate_kwargs,
                    tr_manager=tr_manager
                ) for i in range(self.n_greedy_ascent_init)]

            x_greedy_ascent, acq_greedy_ascent = zip(*exhaustive_ls_return_values)

            # Grab a previously unseen point
            x_greedy_ascent = torch.stack(x_greedy_ascent).cpu()
            acq_greedy_ascent = torch.tensor(acq_greedy_ascent)

            indices = acq_greedy_ascent.argsort()

        for idx in indices:  # Attempt to grab a point from the suggested points
            if not torch.all(x_greedy_ascent[idx] == x_observed, dim=1).any():
                x_next = x_greedy_ascent[idx:idx + 1]
                break

        if x_next is None:  # Attempt to grab a neighbour of the suggested points
            for idx in indices:
                if tr_manager and hamming_distance(x1=tr_manager.center[self.search_space.nominal_dims].to(x), x2=x,
                                                   normalize=False) >= tr_manager.get_nominal_radius():
                    neighbours = cartesian_neighbors_center_attracted(x_greedy_ascent[idx].long(),
                                                                      self.adjacency_mat_list,
                                                                      x_center=tr_manager.center)
                else:
                    neighbours = cartesian_neighbors(x_greedy_ascent[idx].long(), self.adjacency_mat_list)
                valid_filter = np.all(self.input_eval_from_transfx(transf_x=neighbours), axis=1)
                neighbours = neighbours[valid_filter]
                for j in range(neighbours.size(0)):
                    if not torch.all(neighbours[j] == x_observed, dim=1).any():
                        x_next = neighbours[j: j + 1]
                        break
                if x_next is not None:
                    break

        if x_next is None:  # Else, suggest a random point
            if tr_manager:
                point_sampler = lambda n_points: self.search_space.inverse_transform(
                    sample_numeric_and_nominal_within_tr(
                        x_centre=x_centre,
                        search_space=self.search_space,
                        tr_manager=tr_manager,
                        n_points=n_points,
                        numeric_dims=self.numeric_dims,
                        discrete_choices=self.discrete_choices,
                        max_n_perturb_num=self.max_n_perturb_num,
                        model=model,
                        return_numeric_bounds=False
                    )
                )
            else:
                point_sampler = self.search_space.sample

            x_next = self.search_space.transform(
                self.sample_input_valid_points(n_points=1, point_sampler=point_sampler)
            )

        return x_next

    def _exhaustive_ls(self, x_init: torch.FloatTensor, acq_func: AcqBase, model: ModelBase,
                       tr_manager: Optional[TrManagerBase],
                       acq_evaluate_kwargs: dict):
        """
        In order to find local minima of an acquisition function, at each vertex,
        it follows the most decreasing edge starting from an initial point
        if self.max_descent is infinity, this method tries to find local maximum, otherwise,
        it may stop at a noncritical vertex (this option is for a computational reason)
        """

        n_ascent = 0
        x = x_init
        assert np.all(self.input_eval_from_transfx(x)), self.search_space.inverse_transform(x)

        min_acq = acq_func(x=x, model=model, **acq_evaluate_kwargs)

        while n_ascent < self.max_n_descent:
            if tr_manager and hamming_distance(tr_manager.center[self.search_space.nominal_dims].to(x), x,
                                               normalize=False) >= tr_manager.get_nominal_radius():
                # To get a neighbour in the TR, need to select a category matching the center category
                x_neighbours = cartesian_neighbors_center_attracted(x.long(), self.adjacency_mat_list,
                                                                    x_center=tr_manager.center).to(x)
            else:
                x_neighbours = cartesian_neighbors(x.long(), self.adjacency_mat_list).to(x)

            valid_filter = np.all(self.input_eval_from_transfx(transf_x=x_neighbours), axis=1)
            x_neighbours = x_neighbours[valid_filter]
            if len(x_neighbours) == 0:
                break

            with torch.no_grad():
                acq_neighbours = acq_func(x=x_neighbours, model=model, **acq_evaluate_kwargs)

            min_neighbour_index = acq_neighbours.argmin()
            min_neighbour_acq = acq_neighbours[min_neighbour_index]

            if min_neighbour_acq < min_acq:
                min_acq = min_neighbour_acq
                x = x_neighbours[min_neighbour_index.item()]
                n_ascent += 1
            else:
                break
        return x, min_acq.item()
