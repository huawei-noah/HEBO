from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()


def add_common_args(parser: ArgumentParser):
    opt_group = parser.add_argument_group("weighted retraining")
    opt_group.add_argument("--seed", type=int, required=True)
    opt_group.add_argument("--query_budget", type=int, required=True)
    opt_group.add_argument("--retraining_frequency", type=int, required=True)
    opt_group.add_argument(
        "--samples_per_model",
        type=int,
        default=1000,
        help="Number of samples to draw after each model retraining",
    )
    opt_group.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    opt_group.add_argument("--result_root", type=str, help="root directory to store results in")
    opt_group.add_argument("--pretrained_model_file", type=str, default=None, help="path to pretrained model to use")
    opt_group.add_argument("--version", type=int, default=None,
                           help="Version of the model (not required if `pretrained_model_file` is specified)")
    opt_group.add_argument("--n_retrain_epochs", type=float, default=1.0)
    opt_group.add_argument("--n_init_retrain_epochs", type=float, default=None,
                           help="None to use n_retrain_epochs, 0.0 to skip init retrain")
    opt_group.add_argument("--lso_strategy", type=str, choices=["opt", "sample", "random_search"], required=True)
    opt_group.add_argument("--random_search_type", type=str, default=None,
                           help="Type of search when random search strategy is chosen (None is uniform)")

    opt_group.add_argument("--overwrite", action='store_true',
                           help="Whether to overwrite existing results (that will be found in result dir) - Default: False")
    return parser


def add_gp_args(parser: ArgumentParser):
    gp_group = parser.add_argument_group("Sparse GP")
    gp_group.add_argument("--n_inducing_points", type=int, default=500)
    gp_group.add_argument("--n_rand_points", type=int, default=8000)
    gp_group.add_argument("--n_best_points", type=int, default=2000)
    gp_group.add_argument("--invalid_score", type=float, default=-4.0)
    gp_group = parser.add_argument_group("Acquisition function maximisation")
    gp_group.add_argument("--acq-func-id", type=str, default='ExpectedImprovement',
                          help="Name of the acquisition function to use (`ExpectedImprovement`, `UpperConfidenceBound`...)")
    gp_group.add_argument("--acq-func-kwargs", type=dict, default={}, help="Acquisition function kwargs")
    gp_group.add_argument("--acq-func-opt-kwargs", type=dict, default={},
                          help="Acquisition function Optimisation kwargs")
    gp_group.add_argument("--q", type=int, default=1, help="Acquisition batch size")
    gp_group.add_argument("--num-restarts", type=int, default=100, help="Number of start points")
    gp_group.add_argument("--raw-initial-samples", type=int, default=1000,
                          help="Number of initial points used to find start points")
    gp_group.add_argument("--num-MC-sample-acq", type=int, default=256,
                          help="Number of samples to use to evaluate posterior distribution")
    return parser
