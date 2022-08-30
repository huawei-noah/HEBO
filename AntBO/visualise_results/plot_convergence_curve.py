import os
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from bo.main import BOExperiments
from task.utils import plot_mean_std
from utilities.config_utils import load_config
import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True, description='Script to plot convergence curves for all methods')
    parser.add_argument('--config', type=str,
                        default='./visualise_results/convergence_curve_config.yaml',
                        help='Path to configuration File')
    args = parser.parse_args()

    config = load_config(args.config)
    nm_plots = len(config['antigens'])
    if nm_plots == 1:
        cols = 1
        rows = 1
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(6.2 * cols, 5 * rows))
    else:
        cols = int(nm_plots / 2)
        rows = int(nm_plots / cols)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(6.2 * cols, 5 * rows))
        axs = axs.reshape(rows, cols)
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)

    f_evals = config['f_evals']
    row, col = 0, 0
    for antigen in config['antigens']:

        ax = axs[row, col] if nm_plots > 1 else axs
        if col == (cols - 1):
            row += 1
            col = 0
        else:
            col += 1
        ax.set_title(f"{antigen} len(11)")
        ax.set_xlabel('Number of evaluations', fontsize=16)
        ax.set_ylabel('Minimum Binding Energy', fontsize=16)
        ax.grid()

        for method in config['methods']:
            binding_energy = []
            function_evaluations = []

            for seed in config['random_seeds']:
                try:
                    if method in ["BO_transformed_overlap", "BO_ssk"]:
                        if "search_strategy" not in config['methods'][method]:
                            config['methods'][method]["search_strategy"] = "local"
                        results = pd.read_csv(
                            os.path.join(
                                BOExperiments.get_path(
                                    save_path=config['results_dir'],
                                    antigen=antigen,
                                    kernel_type=config['methods'][method]['kernel_name'],
                                    seed=seed,
                                    cdr_constraints=True,
                                    seq_len=config['sequence_length'],
                                    search_strategy=config['methods'][method]['search_strategy']
                                ), "results.csv"
                            )
                        )
                    else:
                        results = pd.read_csv(os.path.join(
                            config['results_dir'], method,
                            f"antigen_{antigen}_kernel_{config['methods'][method]['kernel_name']}_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}",
                            'results.csv'))

                    binding_energy.append(results['BestValue'].to_numpy()[:])
                    function_evaluations.append(results['Index'].to_numpy()[:])
                except:
                    continue

            if method == 'RealData':
                try:
                    filenames = glob.glob(f"{config['methods'][method]['antigen_path']}/{antigen}/*.txt")
                    for i in range(len(filenames)):
                        if i == 0:
                            sequences = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                        else:
                            sequences.append(pd.read_csv(filenames[i], skiprows=1, sep='\t'))
                    min_energy = sequences['Energy'].values.min()
                    ax.plot(list(range(0, f_evals)), f_evals * [min_energy],
                            color=config['methods'][method]['line_color'],
                            label=config['methods'][method]['label'])
                except:
                    pass
            else:
                if len(binding_energy) == 0:
                    continue
                binding_energy = np.array(binding_energy)
                function_evaluations = np.array(function_evaluations)
                function_evaluations += 1  # So the graph starts from 1

                if len(binding_energy) < f_evals:
                    binding_energy_copy = binding_energy.copy()
                    for i, energy in enumerate(binding_energy):
                        binding_energy_copy[i, np.where(np.isnan(energy))[0][1:]] = np.nanmin(energy)
                    binding_energy = binding_energy_copy.copy()

                ax.plot(list(range(0, f_evals)), np.mean(binding_energy, axis=0)[:f_evals],
                        color=config['methods'][method]['line_color'],
                        label=config['methods'][method]['label'])
                ax.fill_between(list(range(0, f_evals)),
                                np.mean(binding_energy, axis=0)[:f_evals] - config['num_std'] * np.std(binding_energy,
                                                                                                       axis=0)[
                                                                                                :f_evals],
                                np.mean(binding_energy, axis=0)[:f_evals] + config['num_std'] * np.std(binding_energy,
                                                                                                       axis=0)[
                                                                                                :f_evals],
                                alpha=0.2, color=config['methods'][method]['line_color'])

    handles, labels = ax.get_legend_handles_labels()
    n_col_legend = 1
    n_bbox = (len(labels) - 1) // n_col_legend + 1
    y_bbox = - 0.02 + n_bbox * 0.014
    lgd = fig.legend(handles, labels, bbox_to_anchor=[0.5, y_bbox], loc='upper center', fancybox=True, shadow=True,
                     ncol=n_col_legend, fontsize=15)
    fig.tight_layout(rect=(0., .10, 1, .1))
    os.makedirs(config['save_dir'], exist_ok=True)
    save_path = os.path.join(config['save_dir'], "binding_energy_vs_funct_evals.pdf")
    plt.savefig(save_path, bbox_extra_artists=(lgd,), pad_inches=0.1, bbox_inches='tight')
    print(f'Saved {os.path.abspath(save_path)}')
    plt.close()
