import argparse

import os
import sys
from pathlib import Path
import warnings
import time

import pandas as pd

warnings.simplefilter('ignore', NumericalWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True, description='Run AntBO')
    parser.add_argument('--antigen', type=str, required=True, help='Name of the target antigen (e.g. 2DD8_S)')
    parser.add_argument('--seq_len', type=int, required=True, help='Longer of the optimized antibody sequence')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of antibodies suggested at each step (default: 1)')
    parser.add_argument('--pre_evals_csv', type=str,
                        help='Path to csv file containing the binding energy of already evaluated antibody sequences.')
    parser.add_argument('--tabular_search_csv', type=str,
                        help='Path to csv file containing the set of eligible antibodies with their pre-computed '
                             'binding energy (to test optimisation in a controlled scenario). '
                             'This table can also have extra columns d1,... dk corresponding '
                             'to vector representation of the antibody')
    parser.add_argument('--path_to_eval_csv', type=str, default="./eval.csv",
                        help='If the black-box evaluations are provided by filling a table, path to this table.')
    parser.add_argument('--cuda_id', type=int, default=0, help='ID of the cuda device to use.')
    parser.add_argument('--seed', type=int, nargs="+", default=[42], help='Seed for reproducibility.')
    parser.add_argument('--absolut_path', type=str, help='Path to Absolut! (if Absolut is needed.)')
    parser.add_argument('--resume', type=int, choices=[0, 1], default=0,
                        help='Whether to resume from an existing run.')
    args = parser.parse_args()

    # TO FILL
    save_path = './results/'
    n_init = 20
    max_iters = 400
    device = f"cuda:{args.cuda_id}"

    # -------- Create config
    config = {
        'pre_evals': args.pre_evals_csv,
        'tabular_search_csv': args.tabular_search_csv,
        'acq': 'ei',
        'ard': True,
        'n_init': n_init,
        'max_iters': max_iters,
        'min_cuda': 10,
        'device': device,
        'seq_len': args.seq_len,
        'normalise': True,
        'batch_size': args.batch_size,
        'save_path': save_path,
        # 'kernel_type': 'mat52',
        # 'kernel_type': 'transformed_overlap',
        'kernel_type': 'rbfBERT',
        'noise_variance': '1e-6',
        'search_strategy': 'local',
        'resume': args.resume,
        # 'bbox': {
        #     'tool': 'Absolut',
        #     'path': args.absolut_path,
        #     'process': 4,
        #     'startTask': 0,
        #     'antigen': args.antigen
        # },
        'bbox': {
            'tool': 'table_filling',
            # 'tool': 'manual',
            'antigen': args.antigen,
            'path_to_eval_csv': args.path_to_eval_csv
        },
    }

    for seed in args.seed:
        start_antigen = time.time()
        boexp = BOExperiments(config=config, cdr_constraints=True, seed=seed)
        boexp.run()
        end_antigen = time.time()
        print(f"Time taken for antigen {args.antigen} = {end_antigen - start_antigen:.1f}s")

        result_dir = boexp.path + f"/results.csv"
        results = pd.read_csv(result_dir, index_col=0)
        results.head()

        print(
            f"Best binder for target antigen {args.antigen}: {results.iloc[-1].BestProtein} "
            f"with binding energy {results.iloc[-1].BestValue:.1f}")
