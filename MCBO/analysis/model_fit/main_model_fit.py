import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from analysis.model_fit.model_fit import get_model_fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description='Get function fit data.')

    parser.add_argument("--task_ids", nargs="+", help="Id of tasks")
    parser.add_argument("--data_method_source", help="Source of (x, y) dataset")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seed of dataset")
    parser.add_argument("--surrogate_ids", nargs="+", help="ID of surrogates to try")
    parser.add_argument("--n_observes", type=int, nargs="+", help="Number of training points to fit the model on.")
    parser.add_argument("--device", type=int, help="CUDA device id")
    parser.add_argument("--absolut_dir", help="Path to Absolut (for antibody task)")

    args = parser.parse_args()

    for task_id in args.task_ids:
        for surrogate_id in args.surrogate_ids:
            for seed in args.seeds:
                for n_observe in args.n_observes:
                    get_model_fit(
                        task_id=task_id,
                        data_method_source=args.data_method_source,
                        surrogate_id=surrogate_id,
                        seed=seed,
                        n_observe=n_observe,
                        device=args.device,
                        absolut_dir=args.absolut_dir
                    )
