""" Code for various 1-off functions """
import argparse
import functools
import glob
import gzip
import os
import pickle
import sys
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.auto import tqdm

# Weighting functions
from utils.utils_plot import get_cummax, get_cummin, plot_mean_std


class DataWeighter:
    weight_types = ["uniform", "rank", "dbas", "fb", "rwr", "cem-pi"]

    def __init__(self, hparams):

        if hparams.weight_type in ["uniform", "fb"]:
            self.weighting_function = DataWeighter.uniform_weights
        elif hparams.weight_type == "rank":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights, k_val=hparams.rank_weight_k
            )

        # Most other implementations are from:
        # https://github.com/dhbrookes/CbAS/blob/master/src/optimization_algs.py
        elif hparams.weight_type == "dbas":
            self.weighting_function = functools.partial(
                DataWeighter.dbas_weights,
                quantile=hparams.weight_quantile,
                noise=hparams.dbas_noise,
            )
        elif hparams.weight_type == "rwr":
            self.weighting_function = functools.partial(
                DataWeighter.rwr_weights, alpha=hparams.rwr_alpha
            )
        elif hparams.weight_type == "cem-pi":
            self.weighting_function = functools.partial(
                DataWeighter.cem_pi_weights, quantile=hparams.weight_quantile
            )

        else:
            raise NotImplementedError

        self.weight_quantile = hparams.weight_quantile
        self.weight_type = hparams.weight_type

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def reduce_weight_variance(weights: np.array, data: np.array):
        """ Reduces the variance of the given weights via data replication """

        weights_new = []
        data_new = []
        for w, d in zip(weights, data):
            if w == 0.0:
                continue
            while w > 1:
                weights_new.append(1.0)
                data_new.append(d)
                w -= 1
            weights_new.append(w)
            data_new.append(d)

        return np.array(weights_new), np.array(data_new)

    @staticmethod
    def uniform_weights(properties: np.array):
        return np.ones_like(properties)

    @staticmethod
    def rank_weights(properties: np.array, k_val: float):
        """
        Calculates rank weights assuming maximization.
        Weights are not normalized.
        """
        if np.isinf(k_val):
            return np.ones_like(properties)
        ranks = np.argsort(np.argsort(-1 * properties))
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights

    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights

    @staticmethod
    def cem_pi_weights(properties: np.array, quantile: float):

        # Find quantile cutoff
        cutoff = np.quantile(properties, quantile)
        weights = (properties >= cutoff).astype(float)
        return weights

    @staticmethod
    def rwr_weights(properties: np.array, alpha: float):

        # Subtract max property value for more stable calculation
        # It doesn't change the weights since they are normalized by the sum anyways
        prop_max = np.max(properties)
        weights = np.exp(alpha * (properties - prop_max))
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def add_weight_args(parser: argparse.ArgumentParser):
        weight_group = parser.add_argument_group("weighting")
        weight_group.add_argument(
            "--weight_type",
            type=str,
            choices=DataWeighter.weight_types,
            required=True,
        )
        weight_group.add_argument(
            "--rank_weight_k",
            type=float,
            default=None,
            help="k parameter for rank weighting",
        )
        weight_group.add_argument(
            "--weight_quantile",
            type=float,
            default=None,
            help="quantile argument for dbas, cem-pi cutoffs",
        )
        weight_group.add_argument(
            "--dbas_noise",
            type=float,
            default=None,
            help="noise parameter for dbas (to induce non-binary weights)",
        )
        weight_group.add_argument(
            "--rwr_alpha", type=float, default=None, help="alpha value for rwr"
        )
        return parser


# Various pytorch functions
def _get_zero_grad_tensor(device):
    """ return a zero tensor that requires grad. """
    loss = torch.as_tensor(0.0, device=device)
    loss = loss.requires_grad_(True)
    return loss


def save_object(obj, filename):
    """ Function that saves an object to a file using pickle """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, "wb") as dest:
        dest.write(result)
    dest.close()


def print_flush(text):
    print(text)
    sys.stdout.flush()


def update_hparams(hparams, model):
    # Make the hyperparameters match
    for k in model.hparams.keys():
        try:
            if vars(hparams)[k] != model.hparams[k]:
                print(
                    f"Overriding hparam {k} from {model.hparams[k]} to {vars(hparams)[k]}"
                )
                model.hparams[k] = vars(hparams)[k]
        except KeyError:  # not all keys match, it's ok
            pass

    # Add any new hyperparameters
    for k in vars(hparams).keys():
        if k not in model.hparams.keys():
            print(f'Adding missing hparam {k} with value "{vars(hparams)[k]}".')
            model.hparams[k] = vars(hparams)[k]


def add_default_trainer_args(parser, default_root=None):
    pl_trainer_grp = parser.add_argument_group("pl trainer")
    pl_trainer_grp.add_argument("--gpu", action="store_true")
    pl_trainer_grp.add_argument("--cuda", type=int, default=None)
    pl_trainer_grp.add_argument("--seed", type=int, default=0)
    pl_trainer_grp.add_argument("--root_dir", type=str, default=default_root)
    pl_trainer_grp.add_argument("--load_from_checkpoint", type=str, default=None)
    pl_trainer_grp.add_argument("--max_epochs", type=int, default=1000)


class SubmissivePlProgressbar(pl.callbacks.ProgressBar):
    """ progress bar with tqdm set to leave """

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Retraining Progress",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


def torch_weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        # init.normal_(m.weight.data)
        init.xavier_uniform_(m.weight.data)

        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        # init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        # init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # init.xavier_normal_(m.weight.data)
        init.xavier_uniform_(m.weight.data)

        # init.normal_(m.bias.data)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # for param in m.parameters():
        #     if len(param.shape) >= 2:
        #         # init.orthogonal_(param.data)
        #         init.orthogonal_(param.data)
        #     else:
        #         # init.normal_(param.data)
        #         init.xavier_uniform_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def get_props(result_path: str, budget: int, maximize: bool, start_score: float, verbose: int) -> List[np.ndarray]:
    all_props: List[np.ndarray] = []

    for result_file in glob.glob(result_path + "/*/results*"):
        if verbose:
            print(result_file.split('/')[-2][-1], end=': ')

        with np.load(result_file) as results:
            props = results['opt_point_properties']
            if props.shape[0] >= budget:
                all_props.append(np.concatenate([[start_score], props[:budget]]))
            if verbose:
                log_files = os.path.join(os.path.dirname(result_path), 'logs.txt')
                best_prop = props.max() if maximize else props.min()
                message = f'{props.shape}: {best_prop:.3f}'
                if os.path.exists(log_files) and props.shape[0] < budget:
                    message += ' - available logs'
                print(message)

    return all_props


def plot_regret(all_props: Union[np.ndarray, List[np.ndarray]], maximize: bool, ax=None, **plot_kw):
    if len(all_props) > 0:

        if ax is None:
            ax = plt.subplot()

        all_props = np.vstack(all_props)

        if maximize:
            data = np.array(get_cummax(all_props))
        else:
            data = np.array(get_cummin(all_props))
        label = plot_kw.pop('label', '')
        label += f" ({len(all_props)} seeds) ({data[:, -1].mean():.2f})"
        alpha = plot_kw.pop('alpha', .3)
        plot_mean_std(data, ax=ax,
                      label=label,
                      markevery=all_props.shape[1] // 10,
                      alpha=alpha, **plot_kw)
    return ax


class RegretPlotter(ABC):

    def __init__(self, budget: int, lso_strategy: str, weight_type: str, k: float, r: int, predict_target: bool,
                 target_predictor_hdims: List[int],
                 metric_loss: str, metric_loss_kw: Dict[str, Any],
                 acq_func_id: str, covar_name: str,
                 input_wp: bool, output_wp: bool,
                 random_search_type: Optional[str],

                 maximise: bool
                 ):

        """

        Args:
            lso_strategy: type of optimisation
            weight_type: type of weighting used for retraining
            k: weighting parameter
            r: period of retraining
            predict_target: whether generative model also predicts target value
            target_predictor_hdims: latent dims of target MLP predictor
            metric_loss: metric loss used to structure embedding space
            metric_loss_kw: kwargs for metric loss
            acq_func_id: name of acquisition function
            covar_name: name of kernel used for the GP
            input_wp: whether input warping is used (Kumaraswarmy)
            output_wp: whether output warping is used
            random_search_type: random search specific strategy
            maximise: whether it is a maximisation of minimisation task
        """
        self.budget = budget
        self.lso_strategy = lso_strategy
        self.weight_type = weight_type
        self.k = k
        self.r = r
        self.predict_target = predict_target
        self.target_predictor_hdims = target_predictor_hdims
        self.metric_loss = metric_loss
        self.metric_loss_kw = metric_loss_kw
        self.acq_func_id = acq_func_id
        self.covar_name = covar_name
        self.input_wp = input_wp
        self.output_wp = output_wp
        self.random_search_type = random_search_type
        self.maximise = maximise

    def plot_regret(self, ax=None, verbose: int = 0, **plot_kw):
        if self.lso_strategy != 'random_search' and self.random_search_type is not None:
            return

        if self.lso_strategy == 'random_search' and (self.input_wp or self.output_wp):
            return

        start_score = self.get_expr_start_score()

        path = self.get_root_path()

        if verbose:
            self.print_exp()
            print(path)

        all_props = get_props(
            result_path=path,
            budget=self.budget,
            maximize=self.maximise,
            start_score=start_score,
            verbose=verbose
        )

        if len(all_props) > 0:
            plot_regret(all_props=all_props, maximize=self.maximise, ax=ax, **plot_kw)
        return ax

    def print_exp(self):
        print(
            self.k, self.r,
            'target-pred' if self.predict_target else '',
            self.metric_loss, self.lso_strategy,
            self.random_search_type if self.lso_strategy == 'random_search' else '', 'inwp' if self.input_wp else '',
            'outwp' if self.output_wp else '',
        )

    @abstractmethod
    def get_expr_start_score(self):
        pass

    @abstractmethod
    def get_root_path(self):
        pass
