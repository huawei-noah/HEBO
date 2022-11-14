import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Generate graph for single antigen
    """
    # antigens = ['4ZFO_F', '1ADQ_A', '5EZO_A', '4OII_A', '4OKV_E', '1NCA_N', '5CZV_A', '5JW4_A']
    antigen = '1ADQ_A'

    random_seeds = [42, 43, 44, 45, 46]
    seq_length = 11

    save_dir = "./ga_final_results/"
    folder_save_format = 'antigen_{}_kernel_GA_seed_{}_cdr_constraint_True_seqlen_{}'

    binding_energy = []
    num_function_evals = []

    for seed in random_seeds:
        results = pd.read_csv(
            os.path.join(save_dir, folder_save_format.format(antigen, seed, seq_length), 'results.csv'))

        binding_energy.append(results['BestValue'].to_numpy())
        num_function_evals.append(results['Index'].to_numpy())

    binding_energy = np.array(binding_energy)
    num_function_evals = np.array(num_function_evals)

    np.save(os.path.join(save_dir, f"binding_energy_{antigen}.npy"), binding_energy)
    np.save(os.path.join(save_dir, f"num_function_evals_{antigen}.npy"), num_function_evals)

    n_std = 1

    plt.figure()
    plt.title(f"Genetic Algorithm {antigen}")
    plt.grid()
    plt.plot(num_function_evals[0], np.mean(binding_energy, axis=0), color="b")
    plt.fill_between(num_function_evals[0],
                     np.mean(binding_energy, axis=0) - n_std * np.std(binding_energy, axis=0),
                     np.mean(binding_energy, axis=0) + n_std * np.std(binding_energy, axis=0),
                     alpha=0.2, color="b")
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Minimum Binding Energy')
    plt.savefig(os.path.join(save_dir, f"binding_energy_vs_funct_evals_{antigen}.png"))
    plt.close()
