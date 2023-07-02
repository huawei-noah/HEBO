import numpy as np
import torch
from typing import Union, Dict, Callable, Optional, List
import pandas as pd

from mcbo.search_space import SearchSpace


def input_eval_from_transfx(transf_x: torch.Tensor, search_space: SearchSpace,
                                    input_constraints: Optional[List[Callable[[Dict], bool]]],
                                    ) -> np.ndarray:
    """
    Evaluate the boolean constraint function on a set of transformed inputs

    Args:
        search_space: optimization search space
        input_constraints: list of funcs taking a point as input and outputting whether the point
                                   is valid or not
    Returns:
        Array of `number of input points \times number of input constraints` booleans
                specifying at index `(i, j)` if input point `i` is valid regarding constraint `j`
    """
    if transf_x.ndim == 1:
        transf_x = transf_x.unsqueeze(0)
    x = search_space.inverse_transform(x=transf_x)
    return input_eval_from_origx(x=x, input_constraints=input_constraints)


def input_eval_from_origx(x: Union[pd.DataFrame, Dict],
                                  input_constraints: Optional[List[Callable[[Dict], bool]]],
                                  ) -> np.ndarray:
    """
    Evaluate the boolean constraint function on a set of non-transformed inputs

    Args:
        x: can contain several input points as a Dataframe, can also be given as a single Dict {var_name: var_value}
        input_constraints: list of funcs taking a point as input and outputting whether the point
                                   is valid or not
    Returns:
        Array of `number of input points \times number of imput constraints` booleans
                specifying at index `(i, j)` if input point `i` is valid regarding constraint function `j`
    """
    if input_constraints is None:
        input_constraints = []
    if isinstance(x, Dict):
        x = [x]
    else:
        x = [x.iloc[i].to_dict() for i in range(len(x))]
    if len(input_constraints) > 0:
        return np.array(
            [[input_constraint(x_) for input_constraint in input_constraints] for x_ in x])
    else:
        return np.ones((len(x), 0)).astype(bool)


def sample_input_valid_points(n_points: int, point_sampler: Callable[[int], pd.DataFrame],
                                      input_constraints: Optional[List[Callable[[Dict], bool]]],
                                      max_trials: int = 100, allow_repeat: bool = True) -> pd.DataFrame:
    """ Get valid points in original space

    Args:
        n_points: number of points desired
        point_sampler: function that can be used to sample points
        input_constraint: function taking a point as input and outputting whether the point is valid or not
        max_trials: max number of trials
        allow_repeat: whether the same point can be suggested several time
    """
    x = point_sampler(n_points)
    if not allow_repeat:
        dup_invalid_filtr = x.duplicated(keep="first").values
        pass_dup_criterion = not np.any(dup_invalid_filtr)
    else:
        pass_dup_criterion = True
    input_constraint_criterion = input_constraints is None or len(input_constraints) == 0
    if pass_dup_criterion and input_constraint_criterion:
        return x

    invalid_input_constr_filtr = np.any(
        1 - input_eval_from_origx(x=x, input_constraints=input_constraints), axis=1
    )
    if allow_repeat:
        dup_invalid_filtr = np.zeros((len(x)))
    else:
        dup_invalid_filtr = x.duplicated(keep="first").values

    invalid_inds = np.arange(n_points)[np.logical_or(invalid_input_constr_filtr, dup_invalid_filtr)]
    trial = 0
    while trial < max_trials * n_points and len(invalid_inds) > 0:
        new_rand_x = point_sampler(len(invalid_inds))
        for i, invalid_ind in enumerate(invalid_inds):
            x.iloc[invalid_ind:invalid_ind + 1] = new_rand_x[i:i + 1]

        if allow_repeat:
            dup_invalid_filtr = np.zeros((len(new_rand_x)))
        else:
            dup_invalid_filtr = new_rand_x.duplicated(keep="first").values

        invalid_input_constr_filtr = np.any(
            1 - input_eval_from_origx(x=new_rand_x, input_constraints=input_constraints),
            axis=1)
        invalid_inds = invalid_inds[np.logical_or(invalid_input_constr_filtr, dup_invalid_filtr)]
        trial += len(invalid_inds)

    if len(invalid_inds) > 0:
        raise RuntimeError("Could not find valid points according to the input boolean constr.")

    return x
