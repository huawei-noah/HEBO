# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import pathlib
from typing import Union, List

import matplotlib
import numpy as np
import pandas as pd
import torch

from mcbo.models import ModelBase
from mcbo.optimizers import OptimizerBase
from mcbo.tasks import TaskBase
from mcbo.utils.results_logger import ResultsLogger

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_model_prediction(model: ModelBase, x: torch.Tensor, y: torch.Tensor, save_path: str):
    assert len(x) == len(y)
    assert x.ndim == 2
    if y.ndim == 2:
        y = y.flatten()

    dtype, device = model.dtype, model.device
    with torch.no_grad():
        mu, var = model.predict(x.to(device, dtype))

    mu = mu.flatten().cpu().numpy()
    std = var.sqrt().flatten().cpu().numpy()

    indices = y.argsort()
    y = y[indices]
    mu = mu[indices]
    std = std[indices]

    # Make a plot of GP fit on training set
    plt.figure()
    plt.title('Model predictions')
    plt.ylabel('y')
    plt.xlabel('Datapoint index')
    plt.grid()
    plt.plot(np.arange(len(mu)), y, 'kx', label='Ground truth')
    plt.plot(np.arange(len(mu)), mu, '--b', label='mean')
    plt.fill_between(np.arange(len(mu)), mu - std, mu + std, color='b', alpha=0.2)
    plt.legend()
    plt.savefig(os.path.join(save_path))
    plt.close()


def plot_convergence_curve(optimizer: Union[OptimizerBase, List[OptimizerBase]], task: TaskBase, save_path: str,
                           plot_per_iter: bool = False):
    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    assert save_path.split('.')[-1] in ['png', 'pdf']
    for optim in optimizer:
        assert len(optim.data_buffer.y) > 0, 'Optimiser has no stored data'

    num_optims = len(optimizer)
    cm = plt.get_cmap('gist_ncar')
    colors = [cm(i // 1 * 1.0 / num_optims) for i in range(num_optims)]

    has_global_optimum = hasattr(task, 'global_optimum')
    y_label = 'Regret' if has_global_optimum else 'f(x*)'
    plt.figure()
    plt.title('Convergence Curve')
    plt.xlabel('Nb BB function evaluations')
    plt.ylabel(y_label)

    for i, optim in enumerate(optimizer):
        # Plot the minimum Black-box function value found up to current iteration
        y, _ = torch.cummin(optim.data_buffer.y.flatten(), dim=0)
        y = y.numpy()
        if has_global_optimum:
            y = y - task.global_optimum

        plt.plot(y, color=colors[i], linestyle='-', label=f'{optim.name} f(x*)')

        # Plot black-box function value found at every iteration
        if plot_per_iter:
            per_iter_y = optim.data_buffer.y.flatten().numpy()
            if has_global_optimum:
                per_iter_y = per_iter_y - task.global_optimum
            plt.plot(per_iter_y, color=colors[i], linestyle='--', alpha=0.5)

    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_single_method_regret(optimizer: OptimizerBase, task: TaskBase, save_path: str):
    assert save_path.split('.')[-1] in ['png', 'pdf']
    assert len(optimizer.y) > 0, 'Optimiser has no stored data'

    plt.figure()
    plt.title(f'{optimizer.name} Convergence Curve')
    plt.xlabel('Nb BB function evals')
    plt.ylabel('Regret')
    regret, _ = torch.cummin(optimizer.y.flatten(), dim=0)
    regret = regret.numpy() - task.global_optimum
    per_iter_regret = optimizer.y.flatten().numpy() - task.global_optimum
    plt.plot(regret, '-b')
    plt.plot(per_iter_regret, '--r', alpha=0.5)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_single_seed_results(save_dir: str, results_logger: ResultsLogger, plotting_settings: dict):
    nb_bb_evals = results_logger.res["nb. bb evals"].to_numpy()
    indices = np.logical_not(np.isnan(nb_bb_evals))

    nb_bb_evals = nb_bb_evals[indices].astype(int)
    fx_star = results_logger.res["f(x*)"].to_numpy()[indices]
    fx = results_logger.res["f(x)"].to_numpy()[indices]

    plt.figure()
    plt.title(f"{plotting_settings.get('optimizer_name')}: Convergence curve")
    plt.grid()
    plt.ylabel("f(x)")
    plt.xlabel("Nb. black-box evaluations")
    plt.xscale(plotting_settings.get("plot_x_scale", "linear"))
    plt.yscale(plotting_settings.get("plot_y_scale", "linear"))
    plt.plot(nb_bb_evals, fx_star, 'b-', label='f(x*)')
    plt.plot(nb_bb_evals, fx, 'b-', alpha=0.3, label='f(x)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'convergence_curve.png'))
    plt.close()


def plot_single_method_results(settings):
    save_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "results",
                             settings.get("task_name"),
                             settings.get("problem_name"), settings.get("save_dir"), "distance_curve.png")
    load_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "results",
                             settings.get("task_name"),
                             settings.get("problem_name"), settings.get("save_dir"), "seed_{}", "results.csv")
    fx = []
    fx_star = []

    for seed in settings.get("random_seeds"):
        results = pd.read_csv(load_path.format(seed))
        fx.append(results["f(x)"].to_numpy()[:])
        fx_star.append(results["f(x*)"].to_numpy()[:])

    nv_bb_evals = results["nb. bb evals"].to_numpy()[:]
    fx = np.array(fx)
    fx_mu = fx.mean(axis=0)
    # fx_std = fx.std(axis=0)

    fx_star = np.array(fx_star)
    fx_star_mu = fx_star.mean(axis=0)
    fx_star_std = fx_star.std(axis=0)

    plt.figure()
    plt.title(f"{settings.get('optimizer_name')}: Convergence Curve")
    plt.grid()
    plt.ylabel("f(x)")
    plt.xlabel("Nb. black-box evaluations")
    plt.xscale(settings.get("plot_x_scale", "linear"))
    plt.yscale(settings.get("plot_y_scale", "linear"))
    plt.plot(nv_bb_evals, fx_star_mu, color="blue", label="f(x*)")
    plt.fill_between(nv_bb_evals, fx_star_mu - fx_star_std, fx_star_mu + fx_star_std, alpha=0.3, color="blue")
    plt.plot(nv_bb_evals, fx_mu, color="red", alpha=0.2, label="mean f(x)")
    # plt.fill_between(nv_bb_evals, fx_mu - fx_star_std, fx_mu + fx_star_std, alpha=0.3, color="red")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_results(settings):
    nb_stds = 1
    markersize = 5
    NUM_METHODS = len(settings['methods'])
    cm = plt.get_cmap('gist_ncar')
    colors = [cm(i // 1 * 1.0 / NUM_METHODS) for i in range(NUM_METHODS)]

    load_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "results",
                             settings.get("task_name"), settings.get("problem_name"), "{}", "seed_{}",
                             "results.csv")

    save_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "results",
                            settings.get("task_name"), settings.get("problem_name"), )

    plt.figure()
    plt.title(f"Convergence Curve (Task: {settings.get('task_name')} - {settings.get('problem_name')})")
    plt.xlabel("Nb. black-box evaluations")
    plt.ylabel("f(x)")
    plt.xscale(settings.get("xscale"))
    plt.yscale(settings.get("yscale"))
    plt.grid()

    for idx, method in enumerate(settings.get("methods")):
        fx = []

        for seed in settings.get("methods").get(f"{method}").get("random_seeds"):
            results = pd.read_csv(load_path.format(method, seed))
            _fx = results["f(x*)"].to_numpy()
            indices = np.logical_not(np.isnan(_fx))
            fx.append(_fx[indices])

        nb_bb_evals = results["nb. bb evals"].to_numpy()[indices]
        fx = np.array(fx)
        fx_mean = fx.mean(axis=0)
        fx_std = fx.std(axis=0)

        plt.plot(nb_bb_evals, fx_mean, color=colors[idx],
                 label=settings.get("methods").get(f"{method}").get("label"),
                 marker=settings.get("methods").get(f"{method}").get("marker"),
                 linestyle=settings.get("methods").get(f"{method}").get("linestyle"),
                 markersize=markersize)

        plt.fill_between(nb_bb_evals, fx_mean - nb_stds * fx_std, fx_mean + nb_stds * fx_std, alpha=0.2,
                         color=colors[idx])

    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
    plt.savefig(os.path.join(save_dir, "convergence_curve.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.title(f"Efficiency Curve (Task: {settings.get('task_name')} - {settings.get('problem_name')})")
    plt.xlabel("Time [s]")
    plt.ylabel("f(x)")
    plt.xscale(settings.get("xscale"))
    plt.yscale(settings.get("yscale"))
    plt.grid()

    for idx, method in enumerate(settings.get("methods")):
        fx = []
        time = []

        for seed in settings.get("methods").get(f"{method}").get("random_seeds"):
            results = pd.read_csv(load_path.format(method, seed))
            _fx = results["f(x*)"].to_numpy()
            indices = np.logical_not(np.isnan(_fx))
            fx.append(_fx[indices])
            time.append(results["time"].to_numpy()[indices])

        time = np.array(time).mean(axis=0)

        fx = np.array(fx)
        fx_mean = fx.mean(axis=0)
        fx_std = fx.std(axis=0)

        plt.plot(time, fx_mean, color=colors[idx],
                 label=settings.get("methods").get(f"{method}").get("label"),
                 marker=settings.get("methods").get(f"{method}").get("marker"),
                 linestyle=settings.get("methods").get(f"{method}").get("linestyle"),
                 markersize=markersize)

        plt.fill_between(time, fx_mean - nb_stds * fx_std, fx_mean + nb_stds * fx_std, alpha=0.2, color=colors[idx])

    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
    plt.savefig(os.path.join(save_dir, "efficiency_curve.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
