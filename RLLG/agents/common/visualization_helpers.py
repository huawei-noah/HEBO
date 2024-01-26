# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.




import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from typing import List, Union, Optional, Tuple


def plot_curves(analysis: ExperimentAnalysis, hps: List[str], metric: str, to_plot: Optional[str] = "final",
                label: Optional[str] = "SAC", n_epochs: Optional[int] = 2000) -> None:
    """
    Plots curves for the specified hyperparameters and metric.

    Parameters:
    -----------
    analysis : tune.ray.ExperimentAnalysis
        Analysis object from Ray Tune.
    hps : List[str]
        List of hyperparameters to choose.
    metric : str
        Metric to plot.
    to_plot : str, optional
        Choose between "final" and "overall" for best final mean or best overall.
        Defaults to "final".
    label : str, optional
        Label for the plotted curve. Defaults to "SAC".
    n_epochs : int, optional
        Number of epochs. Defaults to 2000.

    Returns:
    -----------
    None
    """
    group_by = [f'config/{hp}' for hp in hps if hp != 'repeat_run'] + ['epoch']
    dfs = analysis.trial_dataframes
    conf = analysis.get_all_configs()
    path = os.path.dirname(list(conf.keys())[0])
    conf = {k: {f'config/{_k}': _v for _k, _v in v.items()} for k, v in conf.items()}
    df = pd.concat([dfs[k].assign(**conf[k]) for k in dfs.keys()])
    group = df.groupby(group_by)
    mean = group.mean()
    std = group.std()

    # if overall or final
    if to_plot == "overall":
        plot_max_idx = mean[metric].idxmax()
        best_dict = {'mean': mean.loc[plot_max_idx], 'std': std.loc[plot_max_idx]}
    else:
        final_mean = mean.xs(n_epochs - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        final_std = std.xs(n_epochs - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        plot_max_idx = final_mean[metric].idxmax()
        best_dict = {'mean': final_mean.loc[plot_max_idx], 'std': final_std.loc[plot_max_idx]}

    # plot it
    idx_but_one = plot_max_idx[:-1]
    plot_mean = mean.loc[(idx_but_one)][metric]
    plot_std = std.loc[(idx_but_one)][metric]

    plt.plot(plot_mean, label=label)
    plt.fill_between(plot_mean.index,
                     plot_mean - plot_std,
                     plot_mean + plot_std,
                     alpha=0.2)


def plot_all(env: str,
             agents: List[str],
             experts: List[str],
             lambda_s_choices: List[str],
             init_path: Optional[str] = "..",
             hps: Optional[List[str]] = ['lambda_s_eps'],
             metric: Optional[str] = "mean_avg_return",
             mode: Optional[str] = "max",
             to_plot: Optional[str] = "final",
             n_epochs: Optional[int] = 2000) -> None:
    """
    Plots curves for different agents, experts, and lambda_s choices.

    Parameters:
    -----------
    env : str
        The environment name.
    agents : List[str]
        List of agent names.
    experts : List[str]
        List of expert names.
    lambda_s_choices : List[str]
        List of lambda_s choices.
    init_path : str, optional
        The initialization path. Defaults to "..".
    hps : List[str], optional
        List of hyperparameters to choose. Defaults to ['lambda_s_eps'].
    metric : str, optional
        Metric to plot. Defaults to "mean_avg_return".
    mode : str, optional
        Mode for metric comparison. Defaults to "max".
    to_plot : str, optional
        Choose between "overall" and "final" for best overall or best final mean. Defaults to "final".
    n_epochs : int, optional
        Number of epochs. Defaults to 2000.

    Returns:
    -----------
    None
    """
    assert to_plot in ["overall", "final"]

    plt.figure(figsize=(8, 6))
    for agent in agents:

        experts_copy = experts.copy()
        if agent == "SAC":
            experts_copy = [experts_copy[0]]

        for expert in experts_copy:

            lambda_s_choices_copy = lambda_s_choices.copy()
            if agent == 'SAC' or agent == 'SwitchedSAC':
                lambda_s_choices_copy = [lambda_s_choices_copy[0]]

            for type_lambda_s in lambda_s_choices_copy:

                # get analysis
                if agent == "SAC":
                    path = os.path.join(init_path, "ray_results", env, agent)
                    label = agent
                elif agent == "SwitchedSAC":
                    path = os.path.join(init_path, "ray_results", env, agent, expert)
                    label = f"{agent}-{expert}"
                else:
                    path = os.path.join(init_path, "ray_results", env, agent, expert, type_lambda_s)
                    label = f"{agent}-{expert}-{type_lambda_s}"

                main = sorted(glob.glob(f"{path}/*"), key=os.path.getmtime)[-1].split('/')[-1]
                experiment_checkpoint_path = os.path.join(path, main)
                analysis = ExperimentAnalysis(experiment_checkpoint_path, default_metric=metric, default_mode=mode)

                # plot one curve
                plot_curves(analysis,
                            hps,
                            metric,
                            to_plot=to_plot,
                            label=label,
                            n_epochs=n_epochs)
                plt.legend()