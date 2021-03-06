from botorch.optim.utils import columnwise_clamp
from torch import Tensor
from typing import Optional, Union

def columnwise_clamp_(
    X: Tensor,
    lower: Optional[Union[float, Tensor]] = None,
    upper: Optional[Union[float, Tensor]] = None,
    raise_on_violation: bool = False,
) -> Tensor:
    r"""Clamp values of a Tensor in column-wise fashion (with support for t-batches).

    Inplace version of `columnwise_clamp`

    Args:
        X: The `b x n x d` input tensor. If 2-dimensional, `b` is assumed to be 1.
        lower: The column-wise lower bounds. If scalar, apply bound to all columns.
        upper: The column-wise upper bounds. If scalar, apply bound to all columns.
        raise_on_violation: If `True`, raise an exception when the elments in `X`
            are out of the specified bounds (up to numerical accuracy). This is
            useful for post-processing candidates generated by optimizers that
            satisfy imposed bounds only up to numerical accuracy.

    Returns:
        The clamped tensor.
    """
    X.data = columnwise_clamp(X, lower, upper, raise_on_violation).data
    return X