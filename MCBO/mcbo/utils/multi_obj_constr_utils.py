from typing import Union, List

import numpy as np
import torch


def is_better_than_current(current_y: torch.Tensor, new_y: torch.Tensor,
                           obj_dims: Union[List[int], np.ndarray],
                           out_constr_dims: Union[List[int], np.ndarray],
                           out_upper_constr_vals: torch.Tensor,
                           ) -> bool:
    """ 
    Check whether new_y is better than current_y
    
    Args:
        current_y: the current best 1D tensor of objective and constraint outputs
        new_y: the candidate 1D tensor of objective and constraint outputs
        obj_dims: dimensions in ys corresponding to objective values to minimize
        out_constr_dims: dimensions in ys corresponding to inequality constraints
        out_upper_constr_vals: values of upper bounds for inequality constraints
        
    Returns:
        is_better: whether new_y corresponds to a better point than current_y
    """
    assert len(obj_dims) == 1
    if current_y is None:
        return True
    if len(out_constr_dims) == 0:
        return new_y[obj_dims] < current_y[obj_dims]

    # Get penalties of current best and new point
    current_penalty = torch.max(
        current_y[out_constr_dims] - out_upper_constr_vals.to(current_y)).item()
    new_penalty = torch.max(
        new_y[out_constr_dims] - out_upper_constr_vals.to(new_y)).item()
    if current_penalty <= 0:  # current is valid: need the new to be valid and better than current best
        return new_penalty <= 0 and new_y[obj_dims] < current_y[obj_dims]

    # Current best is not valid: need the new to be more valid or equally valid and better than current best
    if new_penalty < current_penalty:
        return True
    return new_penalty == current_penalty and new_y[obj_dims] < current_y[obj_dims]


def get_valid_filter(y: torch.Tensor,
                     out_constr_dims: Union[List[int], np.ndarray],
                     out_upper_constr_vals: torch.Tensor,
                     ) -> torch.Tensor:
    """ 
    Get boolean tensor specifying whether each entry of `y` fulfill the constraints
    
    Args:
        y: 2D tensor of evaluations
        out_constr_dims: dimensions in ys corresponding to inequality constraints
        out_upper_constr_vals: values of upper bounds for inequality constraints
    """
    if len(out_constr_dims) == 0:
        return torch.ones(len(y)).to(bool)
    return torch.all(y[:, out_constr_dims] <= out_upper_constr_vals.to(y), axis=1)


def get_best_y_ind(y: torch.Tensor,
                   obj_dims: Union[List[int], np.ndarray],
                   out_constr_dims: Union[List[int], np.ndarray],
                   out_upper_constr_vals: torch.Tensor,
                   ) -> int:
    """ 
    Get index of best entry in y, taking into account objective and constraints
    If some entries fulfill the constraints, return the best among them
    Otherwise, return the index of the entry that is the closest to fulfillment
    
    Args:
        y: 2D tensor of evaluations
        obj_dims: dimensions in ys corresponding to objective values to minimize
        out_constr_dims: dimensions in ys corresponding to inequality constraints
        out_upper_constr_vals: values of upper bounds for inequality constraints  
    
    Returns:
          best_ind: index at which best y is observed      
    """
    assert len(obj_dims) == 1, obj_dims
    assert len(y) > 0, y

    filtr_nan = torch.isnan(y).sum(-1) == 0
    if not torch.any(filtr_nan):
        return 0
    remaining_inds = torch.arange(len(filtr_nan), device=y.device)[filtr_nan]

    if len(remaining_inds) == 0:
        return 0
    if len(remaining_inds) == 1:
        return remaining_inds[0]

    y = y[remaining_inds]

    if len(out_constr_dims) == 0:  # just take the best according to observed objective value
        best_ind = y[:, obj_dims].flatten().argmin().item()

    else:
        # get array filtering valid points
        valids = get_valid_filter(y=y, out_constr_dims=out_constr_dims, out_upper_constr_vals=out_upper_constr_vals)

        # There are valid inputs
        if valids.sum() > 0:
            # take best among valid points
            best_valid_ind = y[valids][:, obj_dims].flatten().argmin().item()
            best_ind = torch.arange(len(y)).to(device=valids.device)[valids][best_valid_ind].item()
        else:
            # No valid inputs, return the "least unvalid point"
            penalties = y[:, out_constr_dims] - out_upper_constr_vals.to(y)
            penalty = penalties.max(dim=1)[0]  # torch max returns tuple: (values, indices)
            least_invalid_value = penalty.min()
            selected_inds = penalty == least_invalid_value
            best_selected_ind = y[selected_inds][:, obj_dims].flatten().argmin().item()
            best_ind = torch.arange(len(y)).to(device=selected_inds.device)[selected_inds][best_selected_ind].item()
    return remaining_inds[best_ind].item()
