import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from task.tools import Absolut
from random_search.random_searches import RandomSearch
from utilities.config_utils import load_config

import matplotlib

matplotlib.use("Agg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Script to visualise the CDR3-antigen binding '
                                                 'for various methods at various timesteps')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration File')
    args = parser.parse_args()
    config = load_config(args.config)

    absolut_config = {"antigen": None,
                      "path": config['absolut_config']['path'],
                      "process": config['absolut_config']['process'],
                      'startTask': config['absolut_config']['startTask']}

    # Create a directory to save all results
    # try:
    #     os.mkdir(config['save_dir'])
    # except:
    #     raise Exception("Save directory already exists. Choose another directory name to avoid overwriting data")

    antigens_file = './dataloader/all_antigens.txt'
    with open(antigens_file) as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]
    print(f'Iterating Over {len(antigens)} Antigens In File {antigens_file} \n {antigens}')

    for antigen in antigens:
        absolut_config['antigen'] = antigen

        # Defining the fitness function
        absolut_binding_energy = Absolut(absolut_config)


        def function(x):
            x = x.astype(int)
            return absolut_binding_energy.energy(x)


        binding_energy = []
        num_function_evals = []

        print(f"\nAntigen: {antigen}")

        for i, seed in enumerate(config['random_seeds']):
            start = time.time()
            print(f"\nRandom seed {i + 1}/{len(config['random_seeds'])}")

            np.random.seed(seed)

            _save_dir = os.path.join(
                config['save_dir'],
                f"antigen_{antigen}_kernel_RS_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")

            if os.path.exists(_save_dir):
                print(
                    f"Done Experiment antigen_{antigen}_kernel_RS_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']} already exists. Skipping to next Experiment")
                continue
            else:
                os.mkdir(_save_dir)

            rs = RandomSearch(function=function, dimension=config['sequence_length'], num_iter=config['rs_num_iter'],
                              batch_size=config['rs_batch_size'], save_dir=_save_dir, convergence_curve=True)

            try:
                results = rs.run()
            except FileNotFoundError:
                continue
