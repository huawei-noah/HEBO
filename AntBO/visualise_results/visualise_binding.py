import argparse
import os
import pandas as pd
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from bo.main import BOExperiments
from task.utils import compute_scores
from utilities.config_utils import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        add_help=True,
        description='Script to visualise the CDR3-antigen binding for various methods at various timesteps'
    )
    parser.add_argument('--config', type=str,
                        default='/home/asif/workspace/antigenbinding/visualise_results/visualise_binding_config.yaml',
                        help='Path to configuration File')
    args = parser.parse_args()
    config = load_config(args.config)

    # Initialise the absolut visualiser
    # absolut_visualiser = AbsolutVisualisation(config=config)

    for antigen in config['antigens']:

        for method in config['methods']:

            results = {}
            for iter_num in config['visualisation_timesteps']:
                results[iter_num] = {}
                results[iter_num]['CDR3s'] = []
                results[iter_num]['binding_energies'] = []
                results[iter_num]['seed'] = []

            # Gather all data required for visualisation
            for seed in config['random_seeds']:
                try:
                    # load the results
                    if method in ["BO_transformed_overlap", "BO_ssk"]:
                        if "search_strategy" not in config['methods'][method]:
                            config['methods'][method]["search_strategy"] = "local"
                        _results = pd.read_csv(
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
                        _results = pd.read_csv(
                            os.path.join(
                                config['results_dir'], method,
                                f"antigen_{antigen}_kernel_{config['methods'][method]['kernel']}"
                                f"_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}",
                                'results.csv'))
                except:
                    continue

                binding_energy = _results['BestValue'].dropna()[:200].min()
                cdr3_seq = _results['BestProtein'].dropna()[:200].values[-1]
                scores = compute_scores([cdr3_seq])

                print(
                    f"\nBest binding for antigen {antigen}, method {method}, and iteration number {iter_num}\n")
                print(
                    f"\n - method: {method} \n - bb evals: {iter_num} \n - antigen: {antigen} \n - CDR3: {cdr3_seq} \n - binding energy: {binding_energy:.4f} \n - hp: {scores['hp'][0]:.4f} \n - charge: {scores['charge'][0]:.4f} \n - instability_index: {scores['stability'][0]:.4f}\n")

                # absolut_visualiser.visualise(antigen=antigen, cdr3=cdr3_seq)
