import matplotlib

from agent.utils.hyperopt_utils import plot_results
from agent.utils.utils import get_agent_root_dir

if __name__ == '__main__':
    matplotlib.use('Agg')  # Use the Agg backend for faster rendering

    plot_details = {
        "no_reflection": {"color": "C0", "marker": "None", "ls": "-"},
        "naive": {"color": "C1", "marker": "None", "ls": "-"},
    }

    root_dir = get_agent_root_dir()
    task_id_list = [
        "abalone",
        "bank-churn",
        "bsd",
        "fstp",
        "higgs-boson",
        "mercedes",
        "obesity-risk",
        "rcaf",
        "rrp",
        "scrabble",
        "sf-crime",
        "srhm",
    ]
    for task_id in sorted(task_id_list):
        experiment_dir = root_dir / 'workspace/hyperopt' / task_id
        plot_results(experiment_dir, plot_details=plot_details)
