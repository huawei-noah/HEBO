# Copyright (c) 2021
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# util.py
# Utilities for the MetaBO framework.
# ******************************************************************

import numpy as np


def create_uniform_grid(domain, N_samples_dim):
    D = domain.shape[0]
    x_grid = []
    for i in range(D):
        x_grid.append(np.linspace(int(domain[i, 0]), int(domain[i, 1]), int(N_samples_dim)))
    X_mesh = np.meshgrid(*x_grid)
    X = np.vstack(X_mesh).reshape((D, -1)).T
    return X, X_mesh


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]


def scale_from_domain_to_unit_square(X, domain):
    # X contains elements in domain, translate and stretch them to lie in unit square
    return (X - domain[:, 0]) / domain.ptp(axis=1)


def get_cube_around(X, diam, domain):
    assert X.ndim == 1
    assert domain.ndim == 2
    cube = np.zeros(domain.shape)
    cube[:, 0] = np.max((X - 0.5 * diam, domain[:, 0]), axis=0)
    cube[:, 1] = np.min((X + 0.5 * diam, domain[:, 1]), axis=0)
    return cube
