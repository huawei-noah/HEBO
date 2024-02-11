import pandas as pd
from pymoo.core.problem import Problem
import numpy as np

SM_BOX_SAMPLING = 'box-sampling'
SM_LINSPACE = "linspace"
SM_UNIFORM = "uniform"


def box_sampling(input_bounds, sample_num, x_loc=0.2, x_scale=0.1, method="uniform", norm=False):
    dim_samples = []
    for ib in input_bounds:
        val_range = ib[1] - ib[0]
        if method == 'uniform':
            if norm:
                x_s = max((x_loc - x_scale), 0.0) * val_range + ib[0]
                x_e = min((x_loc + x_scale), 1.0) * val_range + ib[0]
                x_i_samples = np.random.random(sample_num) * (x_e - x_s) + x_s
            else:
                x_s = max(ib[0], x_loc - x_scale)
                x_e = min(ib[1], x_loc + x_scale)
                x_i_samples = np.random.uniform(x_s, x_e, sample_num)
        elif method == 'gaussian':
            if norm:
                _samples = np.random.normal(loc=x_loc, scale=x_scale, size=sample_num)
                _samples = np.clip(_samples, 0.0, 1.0)
                x_i_samples = _samples * val_range + ib[0]
            else:
                _samples = np.random.normal(loc=x_loc, scale=x_scale, size=sample_num)
                x_i_samples = np.clip(_samples, ib[0], ib[1])
        else:
            raise ValueError("Unsupported rnd method:", method)
        dim_samples.append(x_i_samples.reshape(-1, 1))
    return np.concatenate(dim_samples, axis=1)


def dominate(x1, x2, minimization=True):
    """
    x1 (Pareto-ly) dominates x2 iff:
    - For all the objectives i, f_i(x1) \leq f_i(x2)
    - For at least one objective j, such that f_j(x1) < f_j(x2)
    """
    x1_flat = (x1 if minimization else -x1).flatten()
    x2_flat = (x2 if minimization else -x2).flatten()

    assert len(x1_flat) == len(x2_flat)

    dom = (
              all([x1_flat[i] <= x2_flat[i] for i in range(len(x1_flat))])
          ) \
          and (
              any([x1_flat[i] < x2_flat[i] for i in range(len(x1_flat))])
          )

    return dom


def non_dominate(x1, x2, minimization=True):
    """
    x1 is non-dominated byx2 iff:
    - For at least one objective j, such that f_j(x1) < f_j(x2)
    """
    x1_flat = (x1 if minimization else -x1).flatten()
    x2_flat = (x2 if minimization else -x2).flatten()

    assert len(x1_flat) == len(x2_flat)

    non_dom = any([x1_flat[i] < x2_flat[i] for i in range(len(x1_flat))])

    return non_dom

