import argparse
import multiprocessing as mp
import shutil
import subprocess as sp
from pathlib import Path

from agent.tasks.hyperopt import HyperOpt
from agent.utils.utils import get_agent_root_dir


def create_arguments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tasks',
        type=str,
        nargs="*",
        default=None,
        required=True,
        help='list of tasks to run'
    )
    parser.add_argument(
        '--reflection-strategy',
        type=str,
        default=None,
        help='the name of the reflection strategy to use'
    )
    parser.add_argument(
        '--llm-name',
        type=str,
        default='llama-3-8B-Instruct',
        help='the name of the fschat config for the main LLM'
    )
    parser.add_argument(
        '--llm-host',
        type=str,
        default='fschat',
        choices=['fschat', 'llm_playground'],
        help='the name of the host for the main LLM'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=10,
        help='number of parallel workers if `--parallel-runs` is active'
    )
    parser.add_argument(
        '--retry-empty-seeds',
        action='store_true',
        default=False,
        help="remove and restart a task for a seed where the workspace is empty"
    )
    return parser


def run_one_task_and_one_seed(
        experiment_dir: Path,
        llm_host: str,
        llm_name: str,
        task_id: str,
        seed: int,
        retry_empty_seeds: bool,
        reflection_strategy: str = None,
) -> None:
    log_dir = experiment_dir / "logs"
    trajectory_file = experiment_dir / "search_space_0" / "optimization_trajectory.csv"

    # check if we can retry this seed if flag activated
    _do_this_seed: bool = False
    if not experiment_dir.exists():
        _do_this_seed = True
    if experiment_dir.exists() and not trajectory_file.exists() and retry_empty_seeds:
        _do_this_seed = True
        print(f"Restart empty: task={task_id}, seed={seed}", flush=True)
        shutil.rmtree(experiment_dir)

    if _do_this_seed:
        try:
            log_dir.mkdir(parents=True)
            command = (
                f'HYDRA_FULL_ERROR=1 '
                f'python -u src/agent/start.py '
                f'task=hyperopt '
                f'method=direct-hyperopt '
                f'llm@agent.llm={llm_host}/{llm_name} '
                f'hydra.run.dir={log_dir} '
                f'task.task_id={task_id} '
                f'task.seed={seed}'
            )
            if reflection_strategy is not None:
                command += f' task.reflection_strategy={reflection_strategy}'

            command_log = log_dir / 'command.txt'
            log_dir.mkdir(exist_ok=True)
            with open(command_log, 'w') as f:
                f.write(command)

            print(f"Running task {task_id} (seed {seed})...", flush=True)

            r = sp.Popen([command], stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
            out, err = r.communicate()
            out, err = out.decode(), err.decode()

            output_log = log_dir / 'output.txt'
            error_log = log_dir / 'error.txt'

            with open(output_log, 'w') as f:
                f.write(out)

            with open(error_log, 'w') as f:
                f.write(err)

            if (len(err) > 0 and 'Error' in str(err)) or (len(out) > 0 and 'Error' in str(out)):
                print(f"{task_id} (seed {seed}) - There was an error, please check `output.txt` and `error.txt`.")
            else:
                print(f"{task_id} (seed {seed}) was setup successfully!", flush=True)

        except (Exception, SystemExit) as e:
            print(f"{task_id} (seed {seed}) failed with the following error:\n{e}", flush=True)

    else:
        print(f"{task_id} with seed {seed} already processed", flush=True)


if __name__ == '__main__':
    args = create_arguments_parser().parse_args()
    workspace_hyperopt_dir = get_agent_root_dir() / './workspace/hyperopt'
    arglist = []
    for seed in range(args.seeds):
        for task_id in args.tasks:
            results_dir = HyperOpt.get_results_path(
                workspace_path=str(workspace_hyperopt_dir / task_id),
                reflection_strategy=args.reflection_strategy
            )
            experiment_dir = Path(results_dir) / f"seed_{seed}"
            arglist.append(
                (
                    experiment_dir,
                    args.llm_host,
                    args.llm_name,
                    task_id,
                    seed,
                    args.retry_empty_seeds,
                    args.reflection_strategy,
                )
            )
    with mp.Pool(processes=args.num_workers) as p:
        p.starmap(run_one_task_and_one_seed, arglist)
