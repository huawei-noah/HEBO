import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent))

if __name__ == '__main__':
    from mcbo.utils.experiment_utils import run_experiment, get_task_from_id, get_opt

    parser = argparse.ArgumentParser(add_help=True, description='MCBO')
    parser.add_argument("--device_id", type=int, default=0, help="Cuda device id (cpu is used if id is negative)")
    parser.add_argument("--task_id", type=str, nargs="+", required=True, help="Name of the task")
    parser.add_argument("--optimizers_ids", type=str, nargs="+", required=True, help="Name of the methods to run")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--result_dir", type=str, default=None, help="Root of the result dir (./results by default)")
    parser.add_argument("--max_num_iter", type=int, default=200, help="Number of acquisitions")
    parser.add_argument("--bo_n_init", type=int, default=20,
                        help="Number of points to acquire before running acquisition with BO")

    # Antigen binding task
    parser.add_argument("--absolut_dir", type=str, default=None, required=False, help="Path to Absolut! executer.")

    args = parser.parse_args()

    dtype_ = torch.float64
    task_ids_ = args.task_id
    for task_id_ in task_ids_:
        task = get_task_from_id(task_id=task_id_, absolut_dir=args.absolut_dir)
        search_space = task.get_search_space(dtype=dtype_)

        bo_n_init_ = args.bo_n_init
        if args.device_id >= 0 and torch.cuda.is_available():
            bo_device_ = torch.device(f'cuda:{args.device_id}')
        else:
            bo_device_ = torch.device("cpu")

        max_num_iter = args.max_num_iter
        random_seeds = args.seeds

        selected_optimizers = []
        for opt_id in args.optimizers_ids:
            opt = get_opt(
                task=task,
                short_opt_id=opt_id,
                bo_n_init=bo_n_init_,
                dtype=dtype_,
                bo_device=bo_device_
            )

            run_experiment(
                task=task,
                optimizers=[opt],
                random_seeds=random_seeds,
                max_num_iter=max_num_iter,
                save_results_every=max_num_iter,
                very_verbose=args.verbose > 1,
                result_dir=args.result_dir
            )
