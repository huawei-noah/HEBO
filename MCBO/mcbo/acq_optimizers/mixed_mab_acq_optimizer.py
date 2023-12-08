# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import warnings
from typing import Optional, List, Callable, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from mcbo.acq_funcs import AcqBase
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.optimizers.non_bo.multi_armed_bandit import MultiArmedBandit
from mcbo.search_space import SearchSpace
from mcbo.search_space.search_space import SearchSpaceSubSet
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.data_buffer import DataBuffer
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.discrete_vars_utils import round_discrete_vars
from mcbo.utils.model_utils import add_hallucinations_and_retrain_model
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class MixedMabAcqOptimizer(AcqOptimizerBase):
    color_1: str = get_color(ind=3, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return MixedMabAcqOptimizer.color_1

    @property
    def name(self) -> str:
        return f"MAB-{self.num_optimizer}"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 batch_size: int = 1,
                 max_n_iter: int = 200,
                 mab_resample_tol: int = 500,
                 n_cand: int = 5000,
                 n_restarts: int = 5,
                 num_optimizer: str = 'sgd',
                 cont_lr: float = 1e-3,
                 cont_n_iter: int = 100,
                 dtype: torch.dtype = torch.float64,
                 ):

        assert search_space.num_dims == search_space.num_cont + search_space.num_disc + \
               search_space.num_nominal + search_space.num_ordinal, \
            'The Mixed MAB acquisition optimizer does not support permutation variables.'

        assert n_cand >= n_restarts, \
            'The number of random candidates must be > number of points selected for gradient based optimization'

        super(MixedMabAcqOptimizer, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            out_constr_dims=out_constr_dims

        )

        self.n_cats = [int(ub + 1) for ub in search_space.nominal_ub]
        self.n_cand = n_cand
        self.n_restarts = n_restarts
        self.num_optimizer = num_optimizer
        self.cont_lr = cont_lr
        self.cont_n_iter = cont_n_iter
        self.batch_size = batch_size

        # Algorithm initialisation
        if search_space.num_numeric > 0:
            seed = np.random.randint(int(1e6))
            self.sobol_engine = SobolEngine(search_space.num_numeric, scramble=True, seed=seed)

        self.mab_search_space = SearchSpaceSubSet(
            search_space=search_space,
            nominal_dims=True,
            ordinal_dims=True,
            dtype=dtype
        )

        self.mab = MultiArmedBandit(
            search_space=self.mab_search_space,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            batch_size=batch_size,
            max_n_iter=max_n_iter,
            noisy_black_box=True,
            resample_tol=mab_resample_tol,
            fixed_tr_manager=None,
            fixed_tr_centre_nominal_dims=search_space.nominal_dims,
            dtype=dtype
        )

        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.cat_dims = np.sort(self.search_space.nominal_dims + self.search_space.ordinal_dims).tolist()

        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)

        self.inverse_mapping = [(self.numeric_dims + self.cat_dims).index(i) for i in range(self.search_space.num_dims)]

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

        assert (self.n_restarts == 0 and self.n_cand >= n_suggestions) or (self.n_restarts >= n_suggestions)

        self.mab.update_fixed_tr_manager(fixed_tr_manager=tr_manager)

        if self.batch_size != n_suggestions:
            warnings.warn('batch_size used for initialising the algorithm is not equal to n_suggestions received by' +
                          ' the acquisition optimizer. If the batch size is known in advance, consider initialising' +
                          ' the acquisition optimizer with the correct batch size for better performance.')

        x_next = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)

        if n_suggestions > 1:
            # create a local copy of the model
            model = copy.deepcopy(model)
            model.search_space = self.search_space
        else:
            model = model

        if (self.search_space.num_nominal + self.search_space.num_ordinal) > 0 and (
                self.search_space.num_cont > 0 or self.search_space.num_disc):

            if self.input_constraints is not None:  # need to adjust the costraint to cat dims

                self.mab.input_constraints = []

                def input_on_cat(x_nom_ord: Dict, i: int) -> bool:
                    x_pd_norm_ord = self.mab_search_space.sample(1)
                    for k in x_pd_norm_ord.columns:
                        x_pd_norm_ord[k] = x_nom_ord[k]
                    for k in self.search_space.ordinal_dims:
                        x_pd_norm_ord[k] = x_nom_ord[k]
                    x_cat_transf = self.mab_search_space.transform(x_pd_norm_ord)[0]
                    x_transf = self.reconstruct_x(x_numeric=x[self.numeric_dims], x_cat=x_cat_transf).unsqueeze(0)
                    x_ = self.search_space.inverse_transform(x_transf)
                    return self.input_constraints[i](x_.iloc[0].to_dict())

                self.mab.input_constraints = [lambda x_nom_ord_, i=i_: input_on_cat(x_nom_ord_, i) for
                                              i_ in
                                              range(len(self.input_constraints))]

            x_cat = self.mab_search_space.transform(self.mab.suggest(n_suggestions))

            x_cat_unique, x_cat_counts = torch.unique(x_cat, return_counts=True, dim=0)

            for idx, curr_x_cat in enumerate(x_cat_unique):

                if len(x_next):
                    # Add the last point to the model and retrain it
                    add_hallucinations_and_retrain_model(model, x_next[-x_cat_counts[idx - 1].item()])

                x_numeric_ = self.optimize_x_numeric(
                    x_cat=curr_x_cat,
                    n_suggestions=x_cat_counts[idx],
                    model=model,
                    acq_func=acq_func,
                    acq_evaluate_kwargs=acq_evaluate_kwargs,
                    tr_manager=tr_manager
                )
                x_cat_ = curr_x_cat * torch.ones((x_cat_counts[idx], curr_x_cat.shape[0]))

                x_next = torch.cat((x_next, self.reconstruct_x(x_numeric_, x_cat_)))

        elif self.search_space.num_cont > 0:
            x_next = torch.cat((x_next, self.optimize_x_numeric(torch.tensor([]), n_suggestions, model, acq_func,
                                                                acq_evaluate_kwargs, tr_manager)))

        elif self.search_space.num_nominal > 0:
            x_next = self.mab_search_space.transform(self.mab.suggest(n_suggestions))

        return x_next

    def optimize_x_numeric(self, x_cat: torch.Tensor, n_suggestions: int, model: ModelBase, acq_func: AcqBase,
                           acq_evaluate_kwargs: dict, tr_manager: Optional[TrManagerBase]):

        if self.search_space.num_nominal == 0:
            x_cat = torch.zeros(size=(0,), dtype=self.dtype)
        assert x_cat.ndim == 1, x_cat.shape

        extended_x_cat = x_cat * torch.ones((self.n_cand, x_cat.shape[0]))

        # Make a copy of the acquisition function if necessary to avoid changing original model parameters
        if n_suggestions > 1:
            model = copy.deepcopy(model)
            model.search_space = self.search_space

        output = torch.zeros((0, self.search_space.num_numeric), dtype=self.dtype)

        for i in range(n_suggestions):

            if len(output) > 0:
                add_hallucinations_and_retrain_model(model, self.reconstruct_x(output[-1], x_cat))

            # Sample x_cont
            if tr_manager is None:
                def point_sampler(n_points: int) -> pd.DataFrame:
                    samples = self.sobol_engine.draw(n=n_points)  # Note that this assumes x in [0, 1]
                    samples = self.reconstruct_x(x_numeric=samples,
                                                 x_cat=x_cat * torch.ones((samples.shape[0], x_cat.shape[0])))
                    return self.search_space.inverse_transform(samples)

                numeric_lb = 0
                numeric_ub = 1
            else:
                _, numeric_lb, numeric_ub = sample_numeric_and_nominal_within_tr(
                    x_centre=tr_manager.center,
                    search_space=self.search_space,
                    tr_manager=tr_manager,
                    n_points=self.n_cand,
                    numeric_dims=self.numeric_dims,
                    discrete_choices=self.discrete_choices,
                    max_n_perturb_num=20,
                    model=None,
                    return_numeric_bounds=True
                )

                def point_sampler(n_points: int) -> pd.DataFrame:
                    samples = sample_numeric_and_nominal_within_tr(
                        x_centre=tr_manager.center,
                        search_space=self.search_space,
                        tr_manager=tr_manager,
                        n_points=n_points,
                        numeric_dims=self.numeric_dims,
                        discrete_choices=self.discrete_choices,
                        max_n_perturb_num=20,
                        model=None,
                        return_numeric_bounds=False
                    )
                    samples[:, self.search_space.nominal_dims] = x_cat
                    return self.search_space.inverse_transform(samples)

            x_cand = self.sample_input_valid_points(n_points=self.n_cand, point_sampler=point_sampler)
            x_numeric_cand = self.search_space.transform(x_cand)[:, self.numeric_dims]

            if self.search_space.num_nominal > 0:
                x_cand = self.reconstruct_x(x_numeric_cand, extended_x_cat)
            else:
                x_cand = self.reconstruct_x(x_numeric_cand, extended_x_cat)

            # Evaluate all random samples
            acq = torch.zeros(len(x_cand)).to(x_cand)
            i_start = 0
            i_end = 500
            with torch.no_grad():
                while i_start < len(x_cand):
                    acq[i_start:i_end] = acq_func(
                        x=x_cand[i_start:i_end], model=model, **acq_evaluate_kwargs
                    )
                    i_start = i_end
                    i_end += 500

            if self.n_restarts > 0:

                x_cont_best = None
                best_acq = None

                x_local_cand = x_cand[acq.argsort()[:self.n_restarts].detach().cpu().numpy()]

                for x_ in x_local_cand:

                    x_numeric_, x_cat_ = x_[self.numeric_dims], x_[self.search_space.nominal_dims]
                    x_numeric_.requires_grad_(True)

                    if self.num_optimizer == 'adam':
                        optimizer = torch.optim.Adam([{"params": x_numeric_}], lr=self.cont_lr)
                    elif self.num_optimizer == 'sgd':
                        optimizer = torch.optim.SGD([{"params": x_numeric_}], lr=self.cont_lr)
                    else:
                        raise NotImplementedError(f'optimizer {self.num_optimizer} is not implemented.')

                    for _ in range(self.cont_n_iter):
                        optimizer.zero_grad()
                        x_cand = self.reconstruct_x(x_numeric_, x_cat_)
                        acq_x = acq_func(x=x_cand, model=model, **acq_evaluate_kwargs)
                        x_numeric_copy = x_numeric_.clone()

                        try:
                            acq_x.backward()
                            optimizer.step()
                        except RuntimeError:
                            print('Exception occurred during backpropagation. NaN encountered?')
                            pass
                        with torch.no_grad():
                            x_numeric_.data = round_discrete_vars(x_numeric_, self.disc_dims_in_numeric,
                                                                  self.discrete_choices)
                            x_numeric_.data = torch.clip(x_numeric_, min=numeric_lb, max=numeric_ub)
                        if not np.all(self.input_eval_from_transfx(
                                self.reconstruct_x(x_numeric=x_numeric_.detach(),
                                                   x_cat=x_cat))):  # reject this new step
                            x_numeric_ = x_numeric_copy
                    x_numeric_.requires_grad_(False)

                    if best_acq is None or acq_x < best_acq:
                        best_acq = acq_x.item()
                        x_cont_best = x_numeric_

            else:
                x_cont_best = x_numeric_cand[acq.argsort()[0]]

            output = torch.cat((output, x_cont_best.unsqueeze(0)))

        return output

    def reconstruct_x(self, x_numeric: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if x_numeric.ndim == x_cat.ndim == 1:
            return torch.cat((x_numeric, x_cat))[self.inverse_mapping]
        else:
            return torch.cat((x_numeric, x_cat), dim=1)[:, self.inverse_mapping]

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        """
        Function used to update the weights of each of the multi-armed bandit agents.

        :param x:
        :param y:
        :param data_buffer:
        :param n_init:
        :param kwargs:
        :return:
        """
        if len(data_buffer) < n_init:
            return
        elif len(data_buffer) == n_init and self.mab_search_space.num_dims > 0:
            x_cat_init = self.mab_search_space.inverse_transform(data_buffer.x[:, self.cat_dims])
            y_init = data_buffer.y.cpu().numpy()
            self.mab.initialize(x_cat_init, y_init)
        elif self.mab_search_space.num_dims > 0:
            x_cat = self.mab_search_space.inverse_transform(x[:, self.cat_dims])
            y = y.cpu().numpy()

            self.mab.observe(x_cat, y)

            return
