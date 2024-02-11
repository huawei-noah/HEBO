import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats


def safe_power_transform(y, scaler=None):
    """
    Apply power transform to the target
    """
    try:
        if scaler is None:
            try:
                if y.min() <= 0:
                    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                    y_scaled = scaler.fit_transform(y)
                else:
                    scaler = PowerTransformer(method='box-cox', standardize=True)
                    y_scaled = scaler.fit_transform(y)
                    if y_scaled.std() < 0.5:
                        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                        y_scaled = scaler.fit_transform(y)
                if y_scaled.std() < 0.5:
                    raise RuntimeError(f'y_min={y.min()}, '
                                       f'y_scaled.std()={y_scaled.std()} is less than 0.5')
            except:
                scaler = None
                raise
        else:
            y_scaled = scaler.transform(y)
    except Exception as e:
        print("[Warn] Safe power transform fails:", e)
        y_scaled = y.copy()
    return y_scaled, scaler


def yeo_johnson_transform(y, scaler=None):
    """
    Apply power transform to the target
    """
    try:
        if scaler is None:
            try:
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                y_scaled = scaler.fit_transform(y)
                if y_scaled.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
            except:
                scaler = None
                raise
        else:
            y_scaled = scaler.transform(y)
    except Exception as e:
        print("[Warn] Safe power transform fails:", e)
        y_scaled = y.copy()
    return y_scaled, scaler


def batch_scaling(scaler, data, inverse_transform=True):
    """
    Scale a batch of data (row by row)
    :param scaler: sklearn.preprocessing.scaler
    :param data: np.array, batch dim = 0
    :param inverse_transform: whether it is an inverse transform
    :return:
    """
    ls = []
    for i in range(data.shape[0]):
        sample = data[i, :]
        scaled = scaler.inverse_transform(sample.reshape(-1, 1)) \
            if inverse_transform else scaler.transform(sample.reshape(-1, 1))
        ls.append(scaled.reshape(1, -1))
    return np.concatenate(ls, axis=0)


def compute_true_input_variance(func, X, input_sigma, sample_num=1000):
    mean_vals = []
    lcb_vals = []
    ucb_vals = []
    for xi in X:
        xi_samples = np.random.normal(xi, input_sigma, size=(sample_num, 1))
        xi_values = func.evaluate(xi_samples)
        _mean = xi_values.mean()
        _std = xi_values.std()
        mean_vals.append(_mean)
        lcb_vals.append(xi_values.max())
        ucb_vals.append(xi_values.min())

    return np.array(mean_vals), np.array(lcb_vals), np.array(ucb_vals)



def estimate_sample_mean_variance(obs_X, obs_Y, all_X, input_std):
    '''
    Given the observed samples, estimate the mean and variance due to input variation.
    '''
    distrib = stats.multivariate_normal(
        np.zeros(shape=(len(input_std))),
        cov=np.diag(input_std ** 2.0)
    )
    weighted_mean_var = []
    for x in all_X:
        _x = x.reshape(1, -1)
        scope = (_x - input_std * 3.0, _x + input_std * 3.0)  # lcb, ucb
        sel_idx = [
            (i and j) for (i, j) in zip(
                (obs_X >= scope[0]).all(axis=1), (obs_X <= scope[1]).all(axis=1)
            )
        ]
        n_support = sum(sel_idx)
        sel_X = obs_X[sel_idx]
        sel_Y = obs_Y[sel_idx]

        if n_support <= 1:
            weighted_mean_var.append((np.nan, np.nan))
        else:
            deltas = abs(sel_X - x)
            weights = (distrib.pdf(deltas) * (
                    np.prod([6 * s for s in input_std]) / n_support)).flatten()
            weighted_mean = (weights.flatten() * sel_Y.flatten()).sum() / weights.sum()
            weighted_var = (weights * (
                    sel_Y.flatten() - weighted_mean) ** 2.0).sum() / weights.sum()
            weighted_mean_var.append((weighted_mean, weighted_var))
    return weighted_mean_var

