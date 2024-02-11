import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

LINE_STYLES = {
    0: {'color': 'k', 'alpha': 0.9, 'ls': '-'},
    1: {'color': 'r', 'alpha': 0.9, 'ls': '-'},
    2: {'color': 'green', 'alpha': 0.9, 'ls': '--'},
    3: {'color': 'royalblue', 'alpha': 0.9, 'ls': '-', 'marker': '+'},
    4: {'color': 'darkgoldenrod', 'alpha': 0.9, 'ls': '-.'},
    5: {'color': 'purple', 'alpha': 0.9, 'ls': (0, (1, 1)), 'lw': 2},
    6: {'color': 'hotpink', 'alpha': 0.9, 'ls': (0, (5, 1)), 'lw': 2},
    7: {'color': 'olivedrab', 'alpha': 0.9, 'ls': (0, (3, 1, 1, 1, 1, 1)), 'lw': 2},
}

AREA_STYLE = {
    0: {'color': 'silver', 'alpha': 0.2, },
    1: {'color': 'salmon', 'alpha': 0.2, },
    2: {'color': 'mediumseagreen', 'alpha': 0.2, },
    3: {'color': 'lightskyblue', 'alpha': 0.2, },
    4: {'color': 'khaki', 'alpha': 0.2, },
    5: {'color': 'plum', 'alpha': 0.2, },
    6: {'color': 'lightpink', 'alpha': 0.2, },
    7: {'color': 'yellowgreen', 'alpha': 0.2, },

}


def plot_contour(xx, yy, vals, filled=False, fig=None, ax=None, tag=""):
    if ax is None:
        fig, ax = plt.subplots()
    if filled:
        cp = ax.contourf(xx, yy, vals.reshape(xx.shape[0], xx.shape[1]))
    else:
        cp = ax.contour(xx, yy, vals.reshape(xx.shape[0], xx.shape[1]))
    fig.colorbar(cp, ax=ax, shrink=0.8)
    ax.set_title(tag)
    return fig, ax


def plot_3d_mesh(xx, yy, vals, fig=None, ax=None, tag=""):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf_crpt = ax.plot_surface(xx, yy, vals.reshape(xx.shape[0], xx.shape[1]),
                                cmap=plt.get_cmap("coolwarm"), alpha=0.7,
                                linewidth=0, antialiased=False)
    fig.colorbar(surf_crpt, ax=ax, shrink=0.8)
    ax.set_title(tag)
    return fig, ax


def plot_line(X, Y, fig=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    h, = ax.plot(X.flatten(), Y.flatten(), **kwargs)
    return fig, ax, h


def plot_auxline(hlines=None, vlines=None, fig=None, ax=None, hline_kw={}, vline_kw={}):
    if ax is None:
        fig, ax = plt.subplots()

    if hlines is not None:
        for l in hlines:
            ax.axhline(l, ls="--", lw=1, color="tab:grey", alpha=0.5, **hline_kw)

    if vlines is not None:
        for v in vlines:
            ax.axvline(v, ls="--", lw=1, color="tab:grey", alpha=0.5, **vline_kw)

    return fig, ax


def scatter(X, Y=None, fig=None, ax=None, **plot_kw):
    if ax is None:
        fig, ax = plt.subplots()
    if X.ndim == 2:
        ax.scatter([v[0] for v in X], [v[1] for v in X], **plot_kw)
    elif X.ndim == 1:
        ax.scatter(X, Y, **plot_kw)

    return fig, ax


def visualize_model_CI(tr_y, tr_pred_mean, tr_pred_lcb, tr_pred_ucb,
                       te_y, te_pred_mean, te_pred_lcb, te_pred_ucb,
                       obj_names=None, sort_y=True,
                       tag="", save_path=None):
    """
    Visualize the confidence interval of model prediction
    """
    n_output = 1 if tr_y.ndim == 1 else tr_y.shape[1]
    nrows = n_output
    ncols = 2
    fig = plt.figure(figsize=(5.5 * ncols, 5.5 * nrows))
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    tr_y = np.atleast_2d(tr_y)
    tr_pred_mean = np.atleast_2d(tr_pred_mean)
    tr_pred_lcb = np.atleast_2d(tr_pred_lcb)
    tr_pred_ucb = np.atleast_2d(tr_pred_ucb)

    te_y = np.atleast_2d(te_y)
    te_pred_mean = np.atleast_2d(te_pred_mean)
    te_pred_lcb = np.atleast_2d(te_pred_lcb)
    te_pred_ucb = np.atleast_2d(te_pred_ucb)
    for i in range(n_output):
        metric_name = f'obj_{i}' if obj_names is None else obj_names[i]
        logy = True if metric_name == "score" else False
        plot_confidence_bar(tr_y[:, i].flatten(),
                            tr_pred_mean[:, i].flatten(),
                            tr_pred_lcb[:, i].flatten(),
                            tr_pred_ucb[:, i].flatten(),
                            sort_y=sort_y, tag="tr", fig=fig, ax=axes[i, 0], logy=logy,
                            aux_hlines=[min(tr_y[:, i].flatten()), ],
                            title=metric_name)
        plot_confidence_bar(te_y[:, i].flatten(),
                            te_pred_mean[:, i].flatten(),
                            te_pred_lcb[:, i].flatten(),
                            te_pred_ucb[:, i].flatten(),
                            sort_y=sort_y, tag="te", fig=fig, ax=axes[i, 1], logy=logy,
                            aux_hlines=[min(tr_y[:, i].flatten()), ],
                            title=metric_name)

    fig.suptitle(f"{tag}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path)

    return fig


def plot_confidence_bar(y_true, pred_mean, pred_lcb, pred_ucb, x=None,
                        sort_y=True, tag="", fig=None, ax=None,
                        aux_hlines=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    sorted_inds = np.argsort(y_true.flatten()) if sort_y else np.arange(len(y_true))
    y_sorted = y_true[sorted_inds]
    pred_mean_sorted = pred_mean[sorted_inds]
    err_sorted = np.concatenate(
        [(pred_mean_sorted - pred_lcb[sorted_inds]).reshape(1, -1),
         (pred_ucb[sorted_inds] - pred_mean_sorted).reshape(1, -1)],
        axis=0
    )
    if x is None:
        x_pos = np.arange(len(sorted_inds))
    else:
        x_pos = x[sorted_inds]
    ax.errorbar(x_pos, pred_mean_sorted.flatten(), yerr=err_sorted,
                capsize=1, fmt='x', markersize=4, alpha=0.4, color='tab:orange',
                ecolor='tab:orange',
                label=f'[{tag}] $\mu$+-$2\sigma$')
    ax.scatter(x_pos, y_sorted.flatten(),
               marker='o', edgecolors='b', s=4, alpha=0.5,
               label=f'[{tag}] y_true')
    if aux_hlines is not None:
        for l in aux_hlines:
            ax.axhline(l, ls="--", lw=1, color="tab:grey", alpha=0.5)

    logy = kwargs.get("logy", False)
    if logy:
        ax.set_yscale("log", base=2)
    ax.set_xlabel("index")
    ax.set_ylabel("val")
    ax.legend()
    return fig, ax


def plot_confidence_region(lcb, ucb, x=None, fig=None, ax=None, label='', **plot_kws):
    if ax is None:
        fig, ax = plt.subplots()
    if x is None:
        x = range(lcb)
    kw = {'color': 'salmon', 'alpha': 0.7, 'label': label}
    kw.update(plot_kws)
    ax.fill_between(x.squeeze(), lcb.squeeze(), ucb.squeeze(), **kw)
    return fig, ax


def plot_model_knowledge(x, y_true, pred_mean, pred_lcb, pred_ucb,
                         tag="", fig=None, ax=None,
                         aux_hlines=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y_true, color='k', alpha=0.5, label=f'[{tag}] y_true')
    ax.plot(x, pred_mean, alpha=0.5, color='tab:orange', label=f'[{tag}] pred')
    ax.fill_between(x, y1=pred_lcb, y2=pred_ucb, alpha=0.5, color="tab:orange")

    if aux_hlines is not None:
        for l in aux_hlines:
            ax.axhline(l, ls="--", lw=1, color="tab:grey", alpha=0.5)

    logy = kwargs.get("logy", False)
    if logy:
        ax.set_yscale("log", base=2)
    yrange = kwargs.get("yrange", None)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_xlabel("index")
    ax.set_ylabel("val")
    ax.legend()
    return fig, ax


def plot_gp_predictions(preds, te_x, gdt_x=None, gdt_y=None, tr_x=None, tr_y=None,
                        fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, (p_name, mean, lcb, ucb, line_style_key, area_style_key) in enumerate(preds):
        # confidence regions
        if area_style_key is None:
            area_style_key = i + 1
        plot_confidence_region(lcb, ucb, te_x, fig, ax, label=f"2std",
                               **AREA_STYLE[area_style_key])

        # means
        if line_style_key is None:
            line_style_key = i + 1
        plot_line(te_x, mean, fig, ax, label=f"mean", **LINE_STYLES[line_style_key])

    # gdt
    if gdt_x is not None and gdt_y is not None:
        plot_line(gdt_x, gdt_y, fig, ax, label='gdt', **LINE_STYLES[0])

    # tr samples
    if tr_x is not None and tr_y is not None:
        scatter(tr_x, tr_y, fig, ax, label='samples', marker="x", c="b", alpha=0.8)

    return fig, ax


def compare_gdt_and_model_prediction(tr_y, tr_pred_mean, te_y, te_pred_mean, obj_names=None,
                                     tag="", save_path=None):
    n_output = 1 if tr_y.ndim == 1 else tr_y.shape[1]
    nrows = n_output
    ncols = 2
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    # plot prediction over training samples
    tr_y = np.atleast_2d(tr_y)
    tr_pred_mean = np.atleast_2d(tr_pred_mean)
    te_y = np.atleast_2d(te_y)
    te_pred_mean = np.atleast_2d(te_pred_mean)
    for i in range(n_output):
        plot_model_prediction(tr_y[:, i].flatten(),
                              tr_pred_mean[:, i].flatten(),
                              tag="tr", fig=fig, ax=axes[i, 0],
                              title=f'obj_{i}' if obj_names is None else obj_names[i],
                              marker="o", edgecolors='b', c="none", alpha=0.5)

        # plot te prediction
        plot_model_prediction(te_y[:, i].flatten(),
                              te_pred_mean[:, i].flatten(),
                              tag="te", fig=fig, ax=axes[i, 1],
                              title=f'obj_{i}' if obj_names is None else obj_names[i],
                              marker="o", edgecolors='r', c="none", alpha=0.5)

    fig.suptitle(f"{tag}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path)

    return fig


def plot_model_prediction(y_true, y_pred, fig=None, ax=None, tag=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    v_min = int(min(min(y_true), min(y_pred))) - 1
    v_max = int(max(max(y_true), max(y_pred))) + 1
    ax.plot([v_min, v_max], [v_min, v_max], ls="--", color="tab:grey", alpha=0.3)
    ax.scatter(x=y_true.flatten(), y=y_pred.flatten(), label=tag, **kwargs)
    ax.set_xlabel("label")
    ax.set_ylabel("pred")
    if tag is not None:
        ax.set_title(tag)
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(v_min, v_max)
    ax.legend()
    if title is not None:
        ax.set_title(title)

    return fig, ax


def visualize_problem(prob, plot_corrupted=False, anotations=None):
    if prob.n_var == 1:
        nrows = 1
        ncols = 2 if plot_corrupted else 1
        fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
        axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False).flatten()
        plot_line(prob.x_coords[0], prob.raw_mesh_vals, fig, axes[0], color="k", label="gdt")
        if anotations is not None:
            for name, (x, y, plot_kwargs) in anotations.items():
                _plot_kwargs = {'label': name, 'marker': 'o', 'color': 'r', 'alpha': 0.7}
                _plot_kwargs.update(plot_kwargs)
                axes[0].scatter(x, y, **plot_kwargs, label=name)

        if plot_corrupted:
            plot_line(prob.x_coords[0], prob.mesh_vals, fig, axes[1], color="r",
                      label="corrupt")
    elif prob.n_var == 2:
        nrows = 2
        ncols = 2 if plot_corrupted else 1
        fig = plt.figure(figsize=(4 * ncols * 1.5, 3 * nrows * 1.5))
        axes = []
        # raw
        ax_i = 1
        ax0 = fig.add_subplot(nrows, ncols, ax_i, projection='3d')
        axes.append(ax0)
        plot_3d_mesh(prob.x_coords[0], prob.x_coords[1], prob.raw_mesh_vals, fig, ax0, "raw")
        if anotations is not None:
            for name, (x, y, plot_kwargs) in anotations.items():
                _plot_kwargs = {'label': name, 'marker': 'o', 'color': 'r', 'alpha': 0.7}
                _plot_kwargs.update(plot_kwargs)
                ax0.scatter(x.squeeze()[0], x.squeeze()[1], y, **plot_kwargs)
        ax_i += 1

        # corrupted
        if plot_corrupted:
            ax1 = fig.add_subplot(nrows, ncols, ax_i, projection='3d')
            axes.append(ax1)
            plot_3d_mesh(prob.x_coords[0], prob.x_coords[1], prob.mesh_vals,
                         fig, ax1, "crpt")
            ax_i += 1

        # contour of raw
        ax2 = fig.add_subplot(nrows, ncols, ax_i)
        axes.append(ax2)
        plot_contour(prob.x_coords[0], prob.x_coords[1], prob.raw_mesh_vals, False, fig, ax2,
                     "raw contour")
        if anotations is not None:
            for name, (x, y, plot_kwargs) in anotations.items():
                _plot_kwargs = {'label': name, 'marker': 'o', 'color': 'r', 'alpha': 0.7}
                _plot_kwargs.update(plot_kwargs)
                ax2.scatter(x.squeeze()[0], x.squeeze()[1], **plot_kwargs)
        ax_i += 1

        # contour of corrupted
        if plot_corrupted:
            ax3 = fig.add_subplot(nrows, ncols, ax_i)
            axes.append(ax3)
            plot_contour(prob.x_coords[0], prob.x_coords[1], prob.mesh_vals, False,
                         fig, ax3, "crpt contour")
            ax_i += 1
    else:
        raise ValueError("Unsupported n_var: ", prob.n_var)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes


def visualize_training_history(tr_history, tag=None):
    df = pd.DataFrame(tr_history)
    # visualize training
    nrows = df.shape[1] - 1
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(6 * 1, 3 * nrows))
    axes = axes.flatten()
    for i, metric_name in enumerate([c for c in df.columns if "step" not in c]):
        axes[i].plot(df['step'].values, df[metric_name].values, label=metric_name)
        if (df[metric_name].max() - df[metric_name].min()) > 100:
            axes[i].set_yscale("log", base=2)
        axes[i].set_ylabel(metric_name)
    if tag is not None:
        fig.suptitle(tag)
    fig.tight_layout()
    plt.show()


def visualize_sampled_model_prediction(
        train_x: np.array = None, train_y: np.array = None,
        gdt_x: np.array = None, gdt_y: np.array = None,
        test_x: np.array = None, test_y_true: np.array = None,
        test_y_pred: np.array = None, tag="", **pred_plot_kwargs):
    fig = plt.figure()
    # tr samples
    if train_x is not None and train_y is not None:
        plt.scatter(x=train_x.flatten(), y=train_y.flatten(), marker="x", color='b', alpha=0.7,
                    label="tr samples", zorder=100)
    # gdt w/o noise
    if gdt_x is not None and gdt_y is not None:
        plt.plot(gdt_x.flatten(), gdt_y.flatten(), color='tab:grey', label='gdt', zorder=99)
    # test samples w/o noise
    if test_y_true is not None:
        for i in range(test_y_true.shape[0]):
            plt.scatter(x=test_x[i].flatten(), y=test_y_true[i].flatten(), s=6, facecolor='none',
                        color='r', alpha=0.3, zorder=1)
    # prediction from sampled models
    if test_y_pred is not None:
        plot_kw = {'alpha': 0.2, 'zorder': 99, 'color': 'tab:orange', }
        plot_kw.update(pred_plot_kwargs)
        for i in range(test_y_pred.shape[0]):
            plt.plot(test_x[i].flatten(), test_y_pred[i].flatten(), **plot_kw)
    plt.legend()
    fig.suptitle(f"{tag}")
    fig.tight_layout()
    return fig


def visualize_opt_history_y(opt_history, objective_names=None):
    """
    visualize an optimization history
    :param opt_history: a list of (step, batch_X, batch_observations)
    :param objective_names: a list of objective names
    :return: fig
    """
    if objective_names is None:
        n_obj = max([h[2].shape[1] for h in opt_history if h[2] is not None])
        objective_names = [f'obj{i}' for i in range(n_obj)]
    else:
        n_obj = len(objective_names)

    # compute best-ever
    best_ever_list = []
    best_ever_val = None
    for h in opt_history:
        step = h[0]
        batch_objs = h[2]
        best_ever_val = np.nanmin(batch_objs, axis=0, keepdims=True) if best_ever_val is None \
            else np.nanmin(np.vstack([best_ever_val, batch_objs]), axis=0, keepdims=True)
        best_ever_list.append((step, best_ever_val.flatten()))

    ncols = n_obj
    nrows = 1
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows)) if ncols > 1 else plt.figure()
    axes = fig.subplots(nrows=1, ncols=n_obj, squeeze=False).flatten()
    for obj_i in range(n_obj):
        axes[obj_i].scatter(
            [h[0] for h in opt_history for _ in range(len(h[2]))],
            [v[obj_i] for h in opt_history for v in h[2]],
            marker='o', facecolor="none", edgecolors='b', s=15, alpha=0.5,
            label=f"samples"
        )
        axes[obj_i].plot(
            [s for s, be in best_ever_list],
            [be[obj_i] for s, be in best_ever_list],
            color='r',
            label=f"best-ever"
        )
        axes[obj_i].set_xlabel("step")
        axes[obj_i].set_ylabel(f"{objective_names[obj_i]}")
    plt.legend()
    fig.tight_layout()
    return fig


def visualize(lines: list = None, scatters: list = None, preds: list = None,
              fig=None, ax=None, tag=""):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plot lines
    if lines is not None:
        for (line_name, line_x, line_y, plot_kw) in lines:
            _plot_kw = {'alpha': 0.7}
            if plot_kw is not None:
                _plot_kw.update(plot_kw)
            plot_line(line_x, line_y, label=line_name, fig=fig, ax=ax, **_plot_kw)

    # plot samples
    if scatters is not None:
        for (scatter_name, scatter_x, scatter_y, plot_kw) in scatters:
            _plot_kw = {'alpha': 0.3, 's': 9}
            if plot_kw is not None:
                _plot_kw.update(plot_kw)
            scatter(scatter_x, scatter_y, label=scatter_name, fig=fig, ax=ax, **plot_kw)

    # pred
    if preds is not None:
        for (pred_name, pred_x, pred_mean, pred_std, mean_plot_kw, std_plot_kw) in preds:
            # CR
            _std_plot_kw = {'alpha': 0.5}
            if std_plot_kw is not None:
                _std_plot_kw.update(std_plot_kw)
            plot_confidence_region(
                pred_mean - pred_std * 2.0,
                pred_mean + pred_std * 2.0,
                x=pred_x, fig=fig, ax=ax, label=f'{pred_name}_CR',
                **_std_plot_kw
            )
            # mean
            _mean_plot_kw = {'alpha': 0.7}
            if mean_plot_kw is not None:
                _mean_plot_kw.update(mean_plot_kw)
            plot_line(pred_x, pred_mean, label=f"{pred_name}_mean", fig=fig, ax=ax, **_mean_plot_kw)

    ax.legend()
    fig.suptitle(tag)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax


def visualize_opt_history_x(opt_history):
    """
    visualize the parameters along the opt history
    :param opt_history: a list of (step, batch_X, batch_observations)
    :return: fig
    """
    steps = []
    xs = []
    for h in opt_history:
        step = h[0]
        batch_X = h[1]
        steps.extend([step, ] * batch_X.shape[0])
        xs.append(batch_X)
    x_df = pd.concat(xs, axis=0, ignore_index=True)
    n_params = x_df.shape[1]

    ncols = 1
    nrows = n_params
    fig = plt.figure(figsize=(8, 3 * nrows))
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharex=True).flatten()
    for param_i in range(n_params):
        axes[param_i].scatter(steps, x_df.iloc[:, param_i], label=x_df.columns[param_i],
                              marker='o', facecolor='none', edgecolor='tab:orange', alpha=0.7)
        axes[param_i].set_xlabel("step")
        axes[param_i].set_ylabel(f"{x_df.columns[param_i]}")
    fig.tight_layout()
    return fig
