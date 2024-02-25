# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

import copy
from typing import Optional, List, Callable, Dict, Union

import numpy as np
import torch

from mcbo.acq_funcs import AcqBase
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.optimizers import RandomSearch
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.model_utils import add_hallucinations_and_retrain_model


class RandomSearchAcqOptimizer(AcqOptimizerBase):
    color_1: str = RandomSearch.get_color_1()

    @staticmethod
    def get_color_1() -> str:
        return RandomSearchAcqOptimizer.color_1

    @property
    def name(self) -> str:
        return f"RS"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 num_samples: int = 300,
                 dtype: torch.dtype = torch.float64,
                 ):
        """

        Args:
            search_space: search space
            input_constraints: constraints on suggested points
            num_samples: number of random samples to use to optimize the acquisition function
            dtype: type of torch tensors
        """
        self.num_samples = num_samples

        self.numeric_dims = search_space.cont_dims + search_space.disc_dims
        self.cat_dims = np.sort(search_space.nominal_dims + search_space.ordinal_dims).tolist()
        self.discrete_choices = get_discrete_choices(search_space)

        super(RandomSearchAcqOptimizer, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            out_constr_dims=out_constr_dims
        )

    def optimize(self, x: torch.Tensor, n_suggestions: int, x_observed: torch.Tensor, model: ModelBase,
                 acq_func: AcqBase, acq_evaluate_kwargs: dict, tr_manager: Optional[TrManagerBase],
                 **kwargs) -> torch.Tensor:
        if n_suggestions == 1:
            return self._optimize(
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
                  n_suggestions: int,
                  x_observed: torch.Tensor,
                  model: ModelBase,
                  acq_func: AcqBase,
                  acq_evaluate_kwargs: dict,
                  tr_manager: Optional[TrManagerBase],
                  ) -> torch.Tensor:

        x_suggest = torch.zeros((n_suggestions, self.search_space.num_dims), dtype=self.dtype)

        rs = RandomSearch(
            search_space=self.search_space,
            input_constraints=self.input_constraints,
            obj_dims=self.obj_dims,
            out_constr_dims=self.out_constr_dims,
            out_upper_constr_vals=self.out_upper_constr_vals,
            fixed_tr_manager=tr_manager,
            store_observations=True,
            dtype=self.dtype
        )

        with torch.no_grad():
            x_next = rs.suggest(self.num_samples)
            y_next = acq_func(
                x=self.search_space.transform(x_next).to(model.dtype),
                model=model,
                **acq_evaluate_kwargs
            ).view(-1, 1).detach().cpu().numpy()
            rs.observe(x_next, y_next)

        # Check if any of the samples was previous unobserved
        insert_ind = 0
        x_rs, y_rs = rs.data_buffer.x, rs.data_buffer.y
        indices = y_rs.flatten().argsort()
        rejected_candidates = []
        for idx in indices:
            x = x_rs[idx]
            if torch.logical_not((x.unsqueeze(0) == x_observed).all(axis=1)).all():
                x_suggest[insert_ind] = x
                insert_ind += 1
                if insert_ind == len(x_suggest):
                    break
            else:
                rejected_candidates.append(x)

        n_remain = n_suggestions - insert_ind
        if n_remain > 0:  # fill suggestion with random valid points
            x_suggest[-n_remain:] = rejected_candidates[:n_remain]

        return x_suggest
