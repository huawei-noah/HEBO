from torch import Tensor


def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats
    coef_val, p_val = stats.spearmanr(pred, target)
    return coef_val


def pearson(pred, target) -> float:
    from scipy import stats
    coef_val, p_val = stats.pearsonr(pred, target)
    return coef_val


def negative_log_likelihood(pred, pred_std, target) -> float:
    """Compute the negative log-likelihood on the validation dataset"""
    from scipy.stats import norm
    import numpy as np
    n = pred.shape[0]
    res = 0.
    for i in range(n):
        res += (
            np.log(norm.pdf(target[i], pred[i], pred_std[i])).sum()
        )
    return -res


def get_dim_info(n_categories):
    dim_info = []
    offset = 0
    for i, cat in enumerate(n_categories):
        dim_info.append(list(range(offset, offset + cat)))
        offset += cat
    return dim_info


def normalize(X: Tensor, bounds: Tensor) -> Tensor:
    r"""Min-max normalize X w.r.t. the provided bounds.

    Args:
        X: `... x d` tensor of data
        bounds: `2 x d` tensor of lower and upper bounds for each of the X's d
            columns.

    Returns:
        A `... x d`-dim tensor of normalized data, given by
            `(X - bounds[0]) / (bounds[1] - bounds[0])`. If all elements of `X`
            are contained within `bounds`, the normalized values will be
            contained within `[0, 1]^d`.

    Example:
        >>> X = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X_normalized = normalize(X, bounds)
    """
    return (X - bounds[0]) / (bounds[1] - bounds[0])
