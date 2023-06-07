import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent)
sys.path.insert(0, ROOT_PROJECT)

import warnings
from gpytorch.utils.warnings import NumericalWarning
from bo.main import BOExperiments
import time

import pandas as pd

warnings.simplefilter('ignore', NumericalWarning)

# TOFILL
save_path = './results/'
absolut_path = "" #TOFILL '/path/to/executable/Absolut/'
antigen = "2DD8_S"
n_init = 20
max_iters = 40
batch_size = 5
device = "cuda:0"
pre_evals_path="" #TOFILL with absolut path to csv of evaluated points "/path/to/evaluated_points.csv"

# -------- Create config

config = {
    'acq': 'ei',
    'ard': True,
    'n_init': n_init,
    'max_iters': max_iters,
    'min_cuda': 10,
    'device': device,
    'seq_len': 11,
    'normalise': True,
    'batch_size': batch_size,
    'save_path': save_path,
    'kernel_type': 'transformed_overlap',
    'noise_variance': '1e-6',
    'search_strategy': 'local',
    'resume': False,
    # 'bbox': {
    #     'tool': 'Absolut',
    #     'path': absolut_path,
    #     'process': 4,
    #     'startTask': 0,
    #     'antigen': antigen
    # },
    'bbox': {
        'tool': 'manual',
        'antigen': antigen
    },
    'pre_evals': pre_evals_path,
}

if __name__ == "__main__":
    for seed in [42]:
        start_antigen = time.time()
        boexp = BOExperiments(config=config, cdr_constraints=True, seed=seed)
        boexp.run()
        end_antigen = time.time()
        print(f"Time taken for antigen {antigen} = {end_antigen - start_antigen:.1f}s")

        result_dir = boexp.path + f"/results.csv"
        results = pd.read_csv(result_dir, index_col=0)
        results.head()

        print(
            f"Best binder for target antigen {antigen}: {results.iloc[-1].BestProtein} "
            f"with binding energy {results.iloc[-1].BestValue:.1f}")
