# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import warnings
from typing import Optional, List, Callable, Dict, Union

import numpy as np
import pandas as pd
import torch

from mcbo.optimizers.optimizer_base import OptimizerNotBO
from mcbo.search_space import SearchSpace
from mcbo.trust_region.tr_manager_base import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.dependant_rounding import DepRound
from mcbo.utils.distance_metrics import hamming_distance
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class MultiArmedBandit(OptimizerNotBO):
    color_1: str = get_color(ind=6, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return MultiArmedBandit.color_1

    @staticmethod
    def get_color() -> str:
        return MultiArmedBandit.get_color_1()

    @property
    def name(self) -> str:
        if self.tr_manager is not None:
            name = 'Tr-based Multi-Armed Bandit'
        else:
            name = 'Multi-Armed Bandit'
        return name

    @property
    def tr_name(self) -> str:
        return "no-tr"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 batch_size: int = 1,
                 max_n_iter: int = 200,
                 noisy_black_box: bool = False,
                 resample_tol: int = 500,
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 fixed_tr_centre_nominal_dims: Optional[List] = None,
                 dtype: torch.dtype = torch.float64,
                 ):

        assert search_space.num_dims == search_space.num_nominal + search_space.num_ordinal, \
            'The Multi-armed bandit optimizer only supports nominal and ordinal variables.'

        self.batch_size = batch_size
        self.max_n_iter = max_n_iter
        self.resample_tol = resample_tol
        self.noisy_black_box = noisy_black_box

        self.n_cats = [int(ub + 1) for ub in search_space.nominal_ub + search_space.ordinal_ub]

        self.best_ube = 2 * max_n_iter / 3  # Upper bound estimate

        self.gamma = []
        for n_cats in self.n_cats:
            if n_cats > batch_size:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats / batch_size) / (
                        (np.e - 1) * batch_size * self.best_ube)))
            else:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats) / ((np.e - 1) * self.best_ube)))

        self.log_weights = [np.zeros(C) for C in self.n_cats]
        self.prob_dist = None

        if fixed_tr_manager is not None:
            assert 'nominal' in fixed_tr_manager.radii, 'Trust Region manager must contain a radius ' \
                                                        'for nominal variables'
            assert fixed_tr_manager.center is not None, 'Trust Region does not have a centre. ' \
                                                        'Call tr_manager.set_center(center) to set one.'
        if fixed_tr_manager is not None:
            assert fixed_tr_centre_nominal_dims is not None
        self.tr_manager = fixed_tr_manager
        self.fixed_tr_centre_nominal_dims = fixed_tr_centre_nominal_dims
        self.tr_center = None if fixed_tr_manager is None else fixed_tr_manager.center[
            self.fixed_tr_centre_nominal_dims].unsqueeze(0)

        super(MultiArmedBandit, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )

    def update_fixed_tr_manager(self, fixed_tr_manager: Optional[TrManagerBase]):
        assert self.fixed_tr_centre_nominal_dims is not None
        self.tr_manager = fixed_tr_manager
        self.tr_center = None if fixed_tr_manager is None else fixed_tr_manager.center[
            self.fixed_tr_centre_nominal_dims].unsqueeze(0)

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:

        self.update_prob_dist()

        def mab_point_sampler(n_points: int) -> pd.DataFrame:
            sample_points = torch.zeros((n_points, self.search_space.num_dims), dtype=self.dtype)
            # Sample all the categorical variables
            for _j, _num_cat in enumerate(self.n_cats):
                # draw a batch here
                if 1 < n_points < _num_cat:
                    _ht = DepRound(self.prob_dist[_j], k=n_points)
                else:
                    _ht = np.random.choice(_num_cat, n_points, p=self.prob_dist[_j])
                # ht_batch_list size: len(self.C_list) x B
                sample_points[:, _j] = torch.tensor(_ht[:], dtype=self.dtype)
            return self.search_space.inverse_transform(x=sample_points)

        x_next = self.search_space.transform(
            self.sample_input_valid_points(n_points=n_suggestions, point_sampler=mab_point_sampler)
        )

        # Project back all point to the trust region centre
        if self.tr_manager is not None:
            hamming_distances = hamming_distance(x_next, self.tr_center, normalize=False)

            for sample_idx, distance in enumerate(hamming_distances):
                if distance > self.tr_manager.get_nominal_radius():
                    # Project x back to the trust region
                    is_valid = False
                    n_trials = 0
                    while not is_valid and n_trials < 3:
                        candidate_proj = x_next[sample_idx].clone()
                        mask = candidate_proj != self.tr_center[0]
                        indices = np.random.choice(np.arange(len(mask))[mask],
                                                   size=distance.item() - self.tr_manager.get_nominal_radius(),
                                                   replace=False)
                        candidate_proj[indices] = self.tr_center[0][indices]
                        if np.all(self.input_eval_from_transfx(transf_x=candidate_proj)):
                            x_next[sample_idx] = candidate_proj
                            is_valid = True
                        n_trials += 1
                    if not is_valid:
                        # sample a valid point in the TR directly
                        point_sampler = lambda n_points: self.search_space.inverse_transform(
                            sample_numeric_and_nominal_within_tr(x_centre=self.tr_center,
                                                                 search_space=self.search_space,
                                                                 tr_manager=self.tr_manager,
                                                                 n_points=n_points,
                                                                 numeric_dims=[],
                                                                 discrete_choices=[],
                                                                 max_n_perturb_num=0,
                                                                 model=None,
                                                                 return_numeric_bounds=False)
                        )
                        x_next[sample_idx] = self.search_space.transform(
                            self.sample_input_valid_points(n_points=1, point_sampler=point_sampler))[0]

        # Eliminate suggestions that have already been observed and all duplicates in the current batch
        for sample_idx in range(n_suggestions):
            tol = 0
            seen = self.was_sample_seen(
                x_next=x_next, sample_idx=sample_idx
            )

            while seen:
                # Resample
                for j, num_cat in enumerate(self.n_cats):
                    ht = np.random.choice(num_cat, 1, p=self.prob_dist[j])
                    x_next[sample_idx, j] = torch.tensor(ht[:], dtype=self.dtype)

                # Project back all point to the trust region centre
                if self.tr_manager is not None:
                    dist = hamming_distance(x_next[sample_idx:sample_idx + 1], self.tr_center, normalize=False)
                    if dist.item() > self.tr_manager.get_nominal_radius():
                        # Project x back to the trust region
                        mask = x_next[sample_idx] != self.tr_center[0]
                        indices = np.random.choice([i for i, x in enumerate(mask) if x],
                                                   size=dist.item() - self.tr_manager.get_nominal_radius(),
                                                   replace=False)
                        x_next[sample_idx][indices] = self.tr_center[0][indices]

                seen = self.was_sample_seen(
                    x_next=x_next, sample_idx=sample_idx
                )
                tol += 1

                if tol > self.resample_tol:
                    warnings.warn(
                        f'Failed to sample a previously unseen sample within {self.resample_tol} attempts. '
                        f'Consider increasing the \'resample_tol\' parameter. Generating a random sample...')
                    if self.tr_manager is not None:
                        point_sampler = lambda n_points: self.search_space.inverse_transform(
                            sample_numeric_and_nominal_within_tr(
                                x_centre=self.tr_center,
                                search_space=self.search_space,
                                tr_manager=self.tr_manager,
                                n_points=n_points,
                                numeric_dims=[],
                                discrete_choices=[],
                                max_n_perturb_num=0,
                                model=None,
                                return_numeric_bounds=False
                            )
                        )
                    else:
                        point_sampler = self.search_space.sample
                    x_next[sample_idx] = self.search_space.transform(
                        self.sample_input_valid_points(n_points=1, point_sampler=point_sampler))[0]

                    seen = False  # Needed to prevent infinite loop

        return self.search_space.inverse_transform(x_next)

    def was_sample_seen(self, x_next, sample_idx) -> bool:

        seen = False

        if len(x_next) > 1:
            # Check if current sample is already in the batch
            if (x_next[sample_idx:sample_idx + 1] == torch.cat((x_next[:sample_idx], x_next[sample_idx + 1:]))).all(
                    dim=1).any():
                seen = True

        # If the black-box is not noisy, check if the current sample was previously observed
        if (not seen) and (not self.noisy_black_box) and (x_next[sample_idx:sample_idx + 1] == self.data_buffer.x).all(
                dim=1).any():
            seen = True

        return seen

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:

        # Transform x and y to torch tensors
        x_transf = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x_transf) == len(y)

        # Add data to all previously observed data and to the trust region manager
        self.data_buffer.append(x_transf, y)

        # update best x and y
        self.update_best(x_transf=x_transf, y=y)

        # Compute the MAB rewards for each of the suggested categories
        mab_rewards = torch.zeros((len(x_transf), self.search_space.num_dims), dtype=self.dtype)

        # Iterate over the batch
        for batch_idx in range(len(x_transf)):
            _x = x_transf[batch_idx]

            # Iterate over all categorical variables
            for dim_dix in range(self.search_space.num_dims):
                indices = self.data_buffer.x[:, dim_dix] == _x[dim_dix]

                # In MAB, we aim to maximise the reward. Comb Opt optimizers minimize reward,
                # hence, take negative of bb values
                rewards = - self.data_buffer.y[indices]

                if len(rewards) == 0:
                    reward = torch.tensor(0., dtype=self.dtype)
                else:
                    reward = rewards.max()

                    # If possible, map rewards to range[-0.5, 0.5]
                    if self.data_buffer.y.max() != self.data_buffer.y.min():
                        reward = 2 * (rewards.max() - (- self.data_buffer.y).min()) / \
                                 ((- self.data_buffer.y).max() - (-self.data_buffer.y).min()) - 1.

                mab_rewards[batch_idx, dim_dix] = reward

        # Update the probability distribution
        for dim_dix in range(self.search_space.num_dims):
            log_weights = self.log_weights[dim_dix]
            num_cats = self.n_cats[dim_dix]
            gamma = self.gamma[dim_dix]
            prob_dist = self.prob_dist[dim_dix]

            x_transf = x_transf.to(torch.long)
            reward = mab_rewards[:, dim_dix]
            nominal_vars = x_transf[:, dim_dix]  # 1xB
            for ii, ht in enumerate(nominal_vars):
                gt_ht_b = reward[ii]
                estimated_reward = 1.0 * gt_ht_b / prob_dist[ht]
                # if ht not in self.S0:
                log_weights[ht] = (log_weights[ht] + (len(mab_rewards) * estimated_reward * gamma / num_cats)).clip(
                    min=-30, max=30)

            self.log_weights[dim_dix] = log_weights

    def restart(self) -> None:
        self._restart()

        self.gamma = []
        for n_cats in self.n_cats:
            if n_cats > self.batch_size:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats / self.batch_size) / (
                        (np.e - 1) * self.batch_size * self.best_ube)))
            else:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats) / ((np.e - 1) * self.best_ube)))

        self.log_weights = [np.zeros(C) for C in self.n_cats]
        self.prob_dist = None

    def set_x_init(self, x: pd.DataFrame):
        # This does not apply to the MAB algorithm
        warnings.warn('set_x_init does not apply to the MAB algorithm')
        pass

    def initialize(self, x: pd.DataFrame, y: np.ndarray):

        # Transform x and y to torch tensors
        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data and to the trust region manager
        self.data_buffer.append(x, y)

        # update best x and y
        if self.best_y is None:
            batch_idx = y.flatten().argmin()
            self.best_y = y[batch_idx, 0].item()
            self._best_x = x[batch_idx: batch_idx + 1]

        else:
            batch_idx = y.flatten().argmin()
            y_ = y[batch_idx, 0].item()

            if y_ < self.best_y:
                self.best_y = y_
                self._best_x = x[batch_idx: batch_idx + 1]

    def update_prob_dist(self) -> None:

        prob_dist = []

        for j in range(len(self.n_cats)):
            weights = np.exp(self.log_weights[j])
            gamma = self.gamma[j]
            norm = float(sum(weights))
            prob_dist.append(list((1.0 - gamma) * (w / norm) + (gamma / len(weights)) for w in weights))

        self.prob_dist = prob_dist
