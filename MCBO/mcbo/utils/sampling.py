# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np


def latin_hypercube(n_pts: int, num_dims):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, num_dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(num_dims):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, num_dims)) / float(2 * n_pts)
    X += pert
    return X


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def univariate_slice_sampling(logp, x0, dtype='float64', width=1.0, max_steps_out=10):
    """
    Univariate Slice Sampling using doubling scheme

    Args:
        logp: numeric(float) -> numeric(float), a log density function
        x0: numeric(float)
        width:
        max_steps_out:

    Returns:
         numeric(float), sampled x1
    """
    assert dtype in ['float32', 'float64']
    for scaled_width in np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]) * width:

        lower = x0 - scaled_width * np.random.rand()
        upper = lower + scaled_width
        llh0 = logp(x0)
        slice_h = np.log(np.random.rand()) + llh0
        llh_record = {}

        # Step Out (doubling)
        steps_out = 0
        logp_lower = logp(lower)
        logp_upper = logp(upper)
        llh_record[float(lower)] = logp_lower
        llh_record[float(upper)] = logp_upper
        while (logp_lower > slice_h or logp_upper > slice_h) and (steps_out < max_steps_out):
            if np.random.rand() < 0.5:
                lower -= (upper - lower)
            else:
                upper += (upper - lower)
            steps_out += 1
            try:
                logp_lower = llh_record[float(lower)]
            except KeyError:
                logp_lower = logp(lower)
                llh_record[float(lower)] = logp_lower
            try:
                logp_upper = llh_record[float(upper)]
            except KeyError:
                logp_upper = logp(upper)
                llh_record[float(upper)] = logp_upper

        # Shrinkage
        start_upper = upper
        start_lower = lower
        n_steps_in = 0
        while not np.isclose(lower, upper):
            x1 = (upper - lower) * np.random.rand() + lower
            llh1 = logp(x1)
            if llh1 > slice_h and univariate_slice_sampling_accept(logp, x0, x1, slice_h, scaled_width, start_lower,
                                                                   start_upper, llh_record):
                return np.float32(x1) if dtype == 'float32' else np.float64(x1)
            else:
                if x1 < x0:
                    lower = x1
                else:
                    upper = x1
            n_steps_in += 1
        # raise RuntimeError('Shrinkage collapsed to a degenerated interval(point)')

    return np.float32(x0) if dtype == 'float32' else np.float64(x0)  # just returning original value


def univariate_slice_sampling_accept(logp, x0, x1, slice_h, width, lower, upper, llh_record):
    acceptance = False
    while upper - lower > 1.1 * width:
        mid = (lower + upper) / 2.0
        if (x0 < mid and x1 >= mid) or (x0 >= mid and x1 < mid):
            acceptance = True
        if x1 < mid:
            upper = mid
        else:
            lower = mid
        try:
            logp_lower = llh_record[float(lower)]
        except KeyError:
            logp_lower = logp(lower)
            llh_record[float(lower)] = logp_lower
        try:
            logp_upper = llh_record[float(upper)]
        except KeyError:
            logp_upper = logp(upper)
            llh_record[float(upper)] = logp_upper
        if acceptance and slice_h >= logp_lower and slice_h >= logp_upper:
            return False
    return True
