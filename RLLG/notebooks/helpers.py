import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from scipy.integrate import simps
from typing import Union, Any, Dict, List, Optional, Tuple


def plot_curves(analysis: tune.ray.ExperimentAnalysis,
                hps: List[str],
                metric: str,
                rolling_mean: float = 0.6,
                set_hyperparam: Dict[str, Any] = {},
                hyperparam_comparison: Optional[str] = None,
                to_plot: Optional[str] = "final",
                label: Optional[str] = "SAC",
                chosen_max: Optional[int] = 1000,
                n_epochs: Optional[int] = 2000,
                retrieve_auc: Optional[bool] = False) -> Optional[Dict[str, Union[float, float]]]:

    """
    Plot learning curves based on the specified analysis, hyperparameters, and metric.

    Parameters:
    ----------
    analysis : tune.ray.ExperimentAnalysis
        The ray tune analysis object containing information about the experiment
    hps : List[str]
        List of hyperparameters to choose for plotting
    metric : str
        The metric to be plotted
    rolling_mean : float, optional
        The alpha value for exponential weighted moving average (default is 0.6).
    set_hyperparam : Dict[str, Any], optional
        Dictionary specifying hyperparameters and their values to set during plotting (default is an empty dictionary).
    hyperparam_comparison : str, optional
        String specifying the hyperparameters to compare during plotting (default is None).
    to_plot : str, optional
        String specifying whether to plot the best final mean or the best overall (default is "final").
    label : str, optional
        Label for the plot (default is "SAC").
    chosen_max : int, optional
        The chosen maximum value for the metric (default is 1000).
    n_epochs : int, optional
        The number of epochs for plotting (default is 2000).
    retrieve_auc : bool, optional
        Boolean indicating whether to retrieve the area under the curve (AUC) and final performance (default is False).

    Returns:
    ----------
    Optional[Dict[str, Union[float, float]]]
        A dictionary containing AUC and final performance if retrieve_auc is True, otherwise None.
    """
    group_by = [f'config/{hp}' for hp in hps if hp != 'repeat_run'] + ['epoch']
    dfs = analysis.trial_dataframes
    conf = analysis.get_all_configs()
    conf = {k: {f'config/{_k}': _v for _k, _v in v.items()} for k, v in conf.items()}
    df = pd.concat([dfs[k].assign(**conf[k]) for k in dfs.keys()])
    group = df.groupby(group_by)
    mean = group.mean()
    std = group.std()

    # if overall or final
    if to_plot == "overall":
        plot_max_idx_init = mean[metric].idxmax()
        best_dict = {'mean': mean.loc[plot_max_idx_init], 'std': std.loc[plot_max_idx_init]}
    else:
        final_mean = mean.xs(chosen_max - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        final_std = std.xs(chosen_max - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        plot_max_idx_init = final_mean[metric].idxmax()
        best_dict = {'mean': final_mean.loc[plot_max_idx_init], 'std': final_std.loc[plot_max_idx_init]}

    # if set_hyperamam
    if set_hyperparam:
        if len(set_hyperparam) == 1:
            key, value = list(set_hyperparam.items())[0]
            idx_to_remove = hps.index(key)
            plot_max_idx = plot_max_idx_init[:idx_to_remove] + plot_max_idx_init[idx_to_remove + 1:]
            plot_max_idx = list(plot_max_idx)
            plot_max_idx.insert(idx_to_remove, value)
            label += f"_{key}_{value}"
        else:
            for key, value in set_hyperparam.items():
                idx_to_remove = hps.index(key)
                plot_max_idx = plot_max_idx_init[:idx_to_remove] + plot_max_idx_init[idx_to_remove + 1:]
                plot_max_idx = list(plot_max_idx)
                plot_max_idx.insert(idx_to_remove, value)
                label += f"_{key}_{value}"
        plot_max_idx = tuple(plot_max_idx)
    else:
        plot_max_idx = plot_max_idx_init

    # if hyperparam comparison
    if hyperparam_comparison is not None:

        hyperparams = hyperparam_comparison.split('-')

        if len(hyperparams) == 1:

            idx_to_remove = hps.index(hyperparam_comparison)
            idx_but_two = plot_max_idx[:idx_to_remove] + plot_max_idx[idx_to_remove + 1:-1]
            hyperparam_list = list(mean.index.get_level_values(idx_to_remove).unique())

            for hyperparam in hyperparam_list:
                idxs = list(idx_but_two).copy()
                idxs.insert(idx_to_remove, hyperparam)
                idxs = tuple(idxs)
                plot_mean = mean.loc[idxs][metric][:n_epochs].ewm(alpha=rolling_mean).mean()
                plot_std = std.loc[idxs][metric][:n_epochs].ewm(alpha=rolling_mean).mean()

                plt.plot(plot_mean, label=f'{label} - {hyperparam_comparison}_{hyperparam}')
                plt.fill_between(plot_mean.index,
                                 plot_mean - plot_std,
                                 plot_mean + plot_std,
                                 alpha=0.2)

        else:
            #
            hyperparam_0_name, hyperparam_1_name = hyperparams
            idx_to_remove_0 = hps.index(hyperparam_0_name)
            idx_to_remove_1 = hps.index(hyperparam_1_name)
            hyperparam_list_0 = list(mean.index.get_level_values(idx_to_remove_0).unique())
            hyperparam_list_1 = list(mean.index.get_level_values(idx_to_remove_1).unique())

            # get idxs
            if idx_to_remove_0 <= idx_to_remove_1:
                idx_but_three = plot_max_idx[:idx_to_remove_0] + plot_max_idx[
                                                                 idx_to_remove_0 + 1:idx_to_remove_1] + plot_max_idx[
                                                                                                        idx_to_remove_1 + 1:-1]
            else:
                idx_but_three = plot_max_idx[:idx_to_remove_1] + plot_max_idx[
                                                                 idx_to_remove_1 + 1:idx_to_remove_0] + plot_max_idx[
                                                                                                        idx_to_remove_0 + 1:-1]

            for hyperparam_0 in hyperparam_list_0:

                for hyperparam_1 in hyperparam_list_1:

                    idxs = list(idx_but_three).copy()
                    if idx_to_remove_0 <= idx_to_remove_1:
                        idxs.insert(idx_to_remove_0, hyperparam_0)
                        idxs.insert(idx_to_remove_1, hyperparam_1)
                    else:
                        idxs.insert(idx_to_remove_1, hyperparam_1)
                        idxs.insert(idx_to_remove_0, hyperparam_0)
                    idxs = tuple(idxs)

                    plot_mean = mean.loc[idxs][metric][:n_epochs].ewm(alpha=rolling_mean).mean()
                    plot_std = std.loc[idxs][metric][:n_epochs].ewm(alpha=rolling_mean).mean()

                    if hyperparam_comparison == 'decay_rate-norm_scale':
                        plt.plot(plot_mean, label=fr'$\delta={hyperparam_0} - \beta_0={hyperparam_1}$')
                    elif hyperparam_comparison == 'decay_rate-phi':
                        plt.plot(plot_mean, label=fr'$\delta={hyperparam_0} - \Phi={hyperparam_1}$')
                    else:
                        raise NotImplementedError(hyperparams)

                    # plt.plot(plot_mean, label=f'{label} - {hyperparam_comparison}_{hyperparam}')
                    plt.fill_between(plot_mean.index,
                                     plot_mean - plot_std,
                                     plot_mean + plot_std,
                                     alpha=0.2)
    else:
        idx_but_one = plot_max_idx[:-1]
        plot_mean = mean.loc[(idx_but_one)][metric][:n_epochs].ewm(alpha=rolling_mean).mean()
        plot_std = std.loc[(idx_but_one)][metric][:n_epochs].ewm(alpha=rolling_mean).mean()

        plt.plot(plot_mean, label=label)
        plt.fill_between(plot_mean.index,
                         plot_mean - plot_std,
                         plot_mean + plot_std,
                         alpha=0.2)

    if retrieve_auc:
        return {
            'auc': simps(mean.loc[(idx_but_one)][metric][:n_epochs], mean.loc[(idx_but_one)][metric][:n_epochs].index),
            'final_perf': mean.loc[(idx_but_one)][metric][:n_epochs].iloc[-1]
        }


def plot_all(env: str,
             agents: List[str],
             experts: List[str],
             rolling_mean: float = 0.6,
             set_hyperparam: Dict[str, Any] = {},
             hyperparam_comparison: Optional[str] = None,
             init_path: Optional[str] = "..",
             hps: Optional[List[str]] = ['betas'],
             metric: Optional[str] = "mean_avg_return",
             mode: Optional[str] = "max",
             to_plot: Optional[str] = "final",
             chosen_max: Optional[int] = 1000,
             n_epochs: Optional[int] = 2000,
             retrieve_auc: Optional[bool] = False) -> Optional[Dict[str, Union[float, float]]]:
    """
    Plot learning curves for multiple agents and experts based on the specified environment.

    Parameters:
    ----------
    env : str
        The environment for which learning curves will be plotted
    agents : List[str]
        List of agent names to be included in the plot
    experts : List[str]
        List of expert names to be included in the plot
    rolling_mean : float, optional
        The alpha value for exponential weighted moving average (default is 0.6)
    set_hyperparam : Dict[str, Any], optional
        Dictionary specifying hyperparameters and their values to set during plotting (default is an empty dictionary)
    hyperparam_comparison : str, optional
        String specifying the hyperparameters to compare during plotting (default is None)
    init_path : str, optional
        The initial path where ray_results are stored (default is "..")
    hps : List[str], optional
        List of hyperparameters to choose for plotting (default is ['betas'])
    metric : str, optional
        The metric to be plotted (default is "mean_avg_return")
    mode : str, optional
        The mode for selecting the best values (default is "max")
    to_plot : str, optional
        String specifying whether to plot the best final mean or the best overall (default is "final")
    chosen_max : int, optional
        The chosen maximum value for the metric (default is 1000)
    n_epochs : int, optional
        The number of epochs for plotting (default is 2000)
    retrieve_auc : bool, optional
        Boolean indicating whether to retrieve the area under the curve (AUC) and final performance (default is False)

    Returns:
    ----------
    Optional[Dict[str, Union[float, float]]]
        A dictionary containing AUC and final performance for each agent-expert combination if retrieve_auc is True,
        otherwise None.
    """
    assert to_plot in ["overall", "final"]

    if retrieve_auc:
        dict_auc = {}

    # plt.figure(figsize=(8, 6))
    for agent in agents:

        experts_copy = experts.copy()
        if agent == "SAC":
            experts_copy = [experts_copy[0]]
            set_hyperparam_ = {}
        else:
            set_hyperparam_ = set_hyperparam.copy()

        for expert in experts_copy:

            # get analysis
            if agent == "SAC":
                path = os.path.join(init_path, "ray_results", env, agent)
                label = agent
            else:
                path = os.path.join(init_path, "ray_results", env, agent, expert)
                label = f"{agent}-{expert}"

            main = sorted(glob.glob(f"{path}/*"), key=os.path.getmtime)[-1].split('/')[-1]
            experiment_checkpoint_path = os.path.join(path, main)

            analysis = ExperimentAnalysis(experiment_checkpoint_path, default_metric=metric, default_mode=mode)

            # plot one curve
            auc = plot_curves(analysis,
                              hps,
                              metric,
                              rolling_mean=rolling_mean,
                              set_hyperparam=set_hyperparam_,
                              hyperparam_comparison=hyperparam_comparison,
                              to_plot=to_plot,
                              label=label,
                              chosen_max=chosen_max,
                              n_epochs=n_epochs,
                              retrieve_auc=retrieve_auc)
            if retrieve_auc:
                dict_auc[f'{agent}_{expert}'] = auc
            plt.legend()
    if retrieve_auc:
        return dict_auc
