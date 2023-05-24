import numpy as np
from matplotlib import transforms

# simple maximize/minimize routines for finite sets with additional masking
def maximize(f, X, mask=None, both=False):
    return minimize(lambda X: -f(X), X, mask, both)

def minimize(f, X, mask=None, both=False):
    """
    Helper function for minimization.

    :param f: function to minimize
    :param X: evaluation points
    :param mask: mask applied to X
    :param both: return both masked and unmasked minimizer.
    :return:
    """
    if not both and not mask is None:
        X = X[mask]

    res = f(X)
    index = np.argmin(res)

    if both:
        X_masked = X[mask]
        res_masked = res[mask]
        index_masked = np.argmin(res_masked)
        return X[index], res[index], X_masked[index_masked], res_masked[index_masked]

    return X[index], res[index]


# helper functions for plotting
def plot_colored_region(axis, x1, x2, color):
    x1, x2 = np.asscalar(x1), np.asscalar(x2)
    trans = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    l, u = 0., 0.03
    axis.fill_between([x1, x2], [l, l], [u, u], color=color, transform=trans, alpha=0.3)

def plot_parameter_changes(axis, parameter_names, xold, xnew, l, u, tr_radius, x0):
    # normalize
    r = u - l
    xold_norm = (xold - l) / r
    x0_norm = (x0 - l) / r
    xnew_norm = (xnew - l) / r
    d = len(xold)

    # plot relative changes
    axis.set_ylim((0, 1))
    axis.set_xlim((-1, d))
    axis.set_ylabel('normalized parameter value')
    axis.set_xticks(range(d))

    if not parameter_names is None and len(parameter_names) > 0:
        axis.set_xticklabels(parameter_names, rotation='vertical')
    else:
        axis.set_xticklabels([f'X_{i}' for i in range(d)])
    w = 0.15

    # plot normalized changes
    for i, (x_start, x_stop, x_0i) in enumerate(zip(xold_norm, xnew_norm, x0_norm)):
        axis.plot([i - w, i + w], [x_start, x_start], color='C0')
        axis.plot([i - w, i + w], [x_stop, x_stop], color='C0')
        axis.plot([i - w, i + w], [x_0i, x_0i], color='C0', linestyle='--')

        # can't plot an arrow of zero length
        if np.abs(x_start - x_stop) < 0.0001:
            continue

        axis.arrow(i, x_start, 0, x_stop - x_start,
                   transform=axis.transData,
                   head_width=0.2,
                   head_length=0.05,
                   fc='C0', ec='C0',
                   length_includes_head=True, overhang=1, antialiased=True)

    axis.bar(range(d), np.abs(xnew - xold) / tr_radius, alpha=0.5, width=2 * w)


def plot_model_changes(axis, y_x0, y_xnew, std_xnew, y_coord):
    d = len(y_coord)

    twinx = axis.twinx()
    twinx.axhline(y_xnew - y_x0, color='C1')
    # twinx.axhline(ucb_xnew - y_x0)

    # axis.bar(range(d), ucb_coord - y_x0)
    twinx.bar(np.arange(d)+0.33, y_coord - y_x0, alpha=0.5, width=0.3, color='C1')

    twinx.set_ylabel('predicted objective increase')
    axis.set_title(f'Expected increase: {y_xnew - y_x0:.2} +- {std_xnew:.2}')

def dimension_setting_helper(max_config, d):
    """
    parses a 'max' config settings,
    e.g. '3', 'd', 'dimension', '4*dimension', '2.5*dimension'
    return int
    """

    if max_config is None:
        return None

    if isinstance(max_config, int):
        return int(max_config)

    if isinstance(max_config, str):
        if max_config.startswith('d'):
            return d

        if '*' in max_config:
            factor, _ = max_config.split('*')
            return round(float(factor)*d)

    raise Exception("Invalid Config")

