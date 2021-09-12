import argparse
import gc
import glob
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.utils.errors import NotPSDError
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from tqdm import tqdm

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from weighted_retraining.weighted_retraining.partial_train_scripts import partial_train_expr
from weighted_retraining.weighted_retraining.robust_opt_scripts.utils import is_robust
from weighted_retraining.weighted_retraining.bo_torch.mo_acquisition import bo_mo_loop

from weighted_retraining.weighted_retraining.expr.expr_data import get_latent_encodings_aux, get_rec_x_error

from utils.utils_cmd import parse_list
from weighted_retraining.weighted_retraining.bo_torch.gp_torch import gp_torch_train, bo_loop, add_gp_torch_args, \
    gp_fit_test
from weighted_retraining.weighted_retraining.bo_torch.utils import put_max_in_bounds
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES

from utils.utils_save import get_storage_root, save_w_pickle, str_dict
from weighted_retraining.weighted_retraining.expr.equation_vae import EquationGrammarModelTorch
from weighted_retraining.weighted_retraining.expr.expr_data_pt import WeightedExprDataset
from weighted_retraining.weighted_retraining.expr.expr_dataset import get_filepath
from weighted_retraining.weighted_retraining.expr.expr_model_pt import EquationVaeTorch

from weighted_retraining.weighted_retraining.robust_opt_scripts.base import add_common_args
from weighted_retraining.weighted_retraining.expr import expr_data
from weighted_retraining.weighted_retraining.utils import print_flush, DataWeighter, SubmissivePlProgressbar
import pytorch_lightning as pl
import weighted_retraining.weighted_retraining.expr.eq_grammar as grammar

MAX_LEN = 15


def retrain_model(model, datamodule: WeightedExprDataset, save_dir: str, version_str: str, num_epochs: int, cuda: int,
                  semi_supervised: Optional[bool] = False):
    # pl._logger.setLevel(logging.CRITICAL)
    train_pbar = SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val", )

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=[cuda] if cuda is not None else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=True,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar, checkpointer],
    )

    # Fit model
    trainer.fit(model, datamodule)


def get_pretrained_model_path(version: int, k, ignore_percentile, good_percentile,
                              predict_target: bool,
                              metric_loss: str, metric_loss_kw: Dict[str, Any],
                              beta_final: float, beta_metric_loss: float, beta_target_pred_loss: float,
                              n_max_epochs: int, latent_dim: int,
                              hdims: List[int] = None):
    """ Get path of directory where models will be stored

        Args:
            latent_dim: latent space dimension
            metric_loss: metric loss used to train the VAE
            metric_loss_kw: kwargs of the metric loss used to train the VAE
            n_max_epochs: total number of epochs the
            version: version of stored model
            k: weight parameter
            ignore_percentile: portion of original equation dataset ignored
            good_percentile: portion of good original equation dataset included
            predict_target: whether generative model also predicts target value
            beta_metric_loss: weight of the metric loss added to the ELBO
            beta_final: weight of the KL in the ELBO
            beta_target_pred_loss: weight of the target prediction loss added to the ELBO
            hdims: latent dims of target MLP predictor
    """
    model_path = os.path.join(partial_train_expr.get_path(
        k=k,
        ignore_percentile=ignore_percentile,
        good_percentile=good_percentile,
        predict_target=predict_target,
        n_max_epochs=n_max_epochs,
        latent_dim=latent_dim, hdims=hdims, metric_loss=metric_loss,
        beta_final=beta_final, beta_metric_loss=beta_metric_loss,
        beta_target_pred_loss=beta_target_pred_loss,
        metric_loss_kw=metric_loss_kw),
        f'lightning_logs/version_{version}/checkpoints/', 'best.ckpt')
    paths = glob.glob(model_path)
    assert len(paths) == 1, model_path
    return paths[0]


def get_root_path(lso_strategy: str, weight_type, k, r, ignore_percentile, good_percentile,
                  predict_target, latent_dim: int, hdims,
                  metric_loss: str, metric_loss_kw: Dict[str, Any],
                  beta_final: float, beta_metric_loss: float,
                  beta_target_pred_loss: float,
                  acq_func_id: str, acq_func_kwargs: Dict[str, Any], covar_name: str, input_wp,
                  random_search_type: Optional[str],
                  n_max_epochs: int,
                  estimate_rec_error: bool, cost_aware_gamma_sched: Optional[str], use_pretrained: bool,
                  semi_supervised: Optional[bool] = False):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        ignore_percentile: portion of original equation dataset ignored
        good_percentile: portion of good original equation dataset included
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        covar_name: name of kernel used for the GP
        input_wp: whether input warping is used (Kumaraswarmy)
        random_search_type: random search specific strategy
        n_max_epochs: max number of epochs on which model has been trained
        estimate_rec_error:  Whether to estimate reconstruction error when new points are acquired
        cost_aware_gamma_sched: schedule for cost-aware acquisition function parameter `gamma`
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        use_pretrained: whether to start from a pretrained VAE

    Returns:
        path to result dir
    """
    result_path = os.path.join(
        get_storage_root(),
        f"logs/opt/expr/{weight_type}/k_{k}/r_{r}")
    exp_spec = f'ignore_perc_{ignore_percentile}-good_perc_{good_percentile}-epochs_{n_max_epochs}'
    if latent_dim != 25:
        exp_spec += f'-z_dim_{latent_dim}'
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
        exp_spec += f'-b_{float(beta_metric_loss):g}'
    exp_spec += f'-bkl_{beta_final}'
    if not use_pretrained:
        exp_spec += '_scratch'
    if semi_supervised:
        exp_spec += "-semi_supervised"

    if lso_strategy == 'opt':
        acq_func_spec = f"{acq_func_id}_{covar_name}{'_inwp_' if input_wp else str(input_wp)}"
        if 'ErrorAware' in acq_func_id and cost_aware_gamma_sched is not None:
            acq_func_spec += f"_sch-{cost_aware_gamma_sched}"
        if len(acq_func_kwargs) > 0:
            acq_func_spec += f'_{str_dict(acq_func_kwargs)}'
        if is_robust(acq_func_id):
            if estimate_rec_error:
                acq_func_spec += "_rec-est_"

        result_path = os.path.join(
            result_path, exp_spec, acq_func_spec
        )

    elif lso_strategy == 'sample':
        raise NotImplementedError('Sample lso strategy not supported')
        # result_path = os.path.join(result_path, exp_spec, f'latent-sample')
    elif lso_strategy == 'random_search':
        base = f'latent-random-search'
        if random_search_type == 'sobol':
            base += '-sobol'
        else:
            assert random_search_type is None, f'{random_search_type} is invalid'
        result_path = os.path.join(result_path, exp_spec, base)
    else:
        raise ValueError(f'{lso_strategy} not supported: try `opt`, `sample`...')
    return result_path


def get_path(lso_strategy: str, weight_type, k, r, ignore_percentile, good_percentile, predict_target,
             latent_dim: int, hdims, metric_loss: str, metric_loss_kw: Dict[str, Any],
             beta_final: float, beta_metric_loss: float, beta_target_pred_loss: float,
             acq_func_id: str, acq_func_kwargs: Dict[str, Any], covar_name: str,
             input_wp, seed, random_search_type: Optional[str], n_max_epochs: int, use_pretrained: bool,
             estimate_rec_error: bool, cost_aware_gamma_sched: Optional[str],
             semi_supervised: Optional[bool] = False):
    """ Get result path

    Args:
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        ignore_percentile: portion of original equation dataset ignored
        good_percentile: portion of good original equation dataset included
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquistion function kwargs
        covar_name: name of kernel used for the GP
        input_wp: whether input warping is used (Kumaraswarmy)
        seed: seed for reproducibility
        random_search_type: random search specific strategy
        n_max_epochs: Total number of training epochs the model has been trained on
        estimate_rec_error:  Whether to estimate reconstruction error when new points are acquired
        cost_aware_gamma_sched: schedule for cost-aware acquisition function parameter `gamma`
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        use_pretrained: whether to start from a pretrained VAE

    Returns:
        path to result dir
    """
    result_path = get_root_path(
        lso_strategy=lso_strategy,
        weight_type=weight_type,
        k=k,
        r=r,
        ignore_percentile=ignore_percentile,
        good_percentile=good_percentile,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        metric_loss=metric_loss,
        metric_loss_kw=metric_loss_kw,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_metric_loss=beta_metric_loss,
        beta_final=beta_final,
        acq_func_id=acq_func_id,
        acq_func_kwargs=acq_func_kwargs,
        covar_name=covar_name,
        input_wp=input_wp,
        random_search_type=random_search_type,
        n_max_epochs=n_max_epochs,
        estimate_rec_error=estimate_rec_error,
        cost_aware_gamma_sched=cost_aware_gamma_sched,
        semi_supervised=semi_supervised,
        use_pretrained=use_pretrained
    )
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)

    parser = add_common_args(parser)
    parser = WeightedExprDataset.add_model_specific_args(parser)
    parser = add_gp_torch_args(parser)
    parser = DataWeighter.add_weight_args(parser)

    parser.add_argument(
        "--ignore_percentile",
        type=int,
        default=50,
        help="percentile of scores to ignore"
    )
    parser.add_argument(
        "--good_percentile",
        type=int,
        default=0,
        help="percentile of good scores selected"
    )
    parser.add_argument(
        '--use_test_set',
        dest="use_test_set",
        action="store_true",
        help="flag to use a test set for evaluating the sparse GP"
    )
    parser.add_argument(
        '--use_full_data_for_gp',
        dest="use_full_data_for_gp",
        action="store_true",
        help="flag to use the full dataset for training the GP"
    )
    parser.add_argument(
        "--n_decode_attempts",
        type=int,
        default=100,
        help="number of decoding attempts",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="cuda ID",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        required=True,
        help="Seed that has been used to generate the dataset"
    )
    parser.add_argument(
        "--input_wp",
        action='store_true',
        help="Whether to apply input warping"
    )
    parser.add_argument(
        "--predict_target",
        action='store_true',
        help="Generative model predicts target value",
    )
    parser.add_argument(
        "--target_predictor_hdims",
        type=list,
        default=None,
        help="Hidden dimensions of MLP predicting target values",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=25,
        help="Hidden dimension the latent space",
    )
    vae_group = parser.add_argument_group("Metric learning")
    vae_group.add_argument(
        "--metric_loss",
        type=str,
        help="Metric loss to add to VAE loss during training of the generative model to get better "
             "structured latent space (see `METRIC_LOSSES`)",
    )
    vae_group.add_argument(
        "--metric_loss_kw",
        type=dict,
        default=None,
        help="Threshold parameter for contrastive loss",
    )
    vae_group.add_argument(
        "--training_max_epochs",
        type=int,
        required=True,
        help="Total number of training epochs the model has been trained on",
    )
    vae_group.add_argument(
        "--estimate_rec_error",
        action='store_true',
        help="Whether to estimate reconstruction error when new points are acquired",
    )
    vae_group.add_argument(
        "--cost_aware_gamma_sched",
        type=str,
        default=None,
        choices=(None, 'fixed', 'linear', 'reverse_linear', 'exponential', 'reverse_exponential', 'post_obj_var',
                 'post_obj_inv_var', 'post_err_var', 'post_err_inv_var', 'post_min_var', 'post_var_tradeoff',
                 'post_var_inv_tradeoff'),
        help="Schedule for error-aware acquisition function parameter `gamma`",
    )
    vae_group.add_argument(
        "--test_gp_error_fit",
        action='store_true',
        help="test the gp fit on the predicted reconstruction error on a validation set",
    )
    vae_group.add_argument(
        "--beta_target_pred_loss",
        type=float,
        default=1.,
        help="Weight of the target_prediction loss added in the ELBO",
    )
    vae_group.add_argument(
        "--beta_metric_loss",
        type=float,
        default=1.,
        help="Weight of the metric loss added in the ELBO",
    )
    vae_group.add_argument(
        "--beta_final",
        type=float,
        help="Weight of the kl loss in the ELBO",
    )
    vae_group.add_argument(
        "--semi_supervised",
        action='store_true',
        help="Start BO from VAE trained with unlabelled data.",
    )
    vae_group.add_argument(
        "--n_init_bo_points",
        type=int,
        default=None,
        help="Number of data points to use at the start of the BO if using semi-supervised training of the VAE."
             "(We need at least SOME data to fit the GP(s) etc.)",
    )
    vae_group.add_argument(
        "--beta_start",
        type=float,
        default=1e-6,
        help="starting beta value; if None then no beta annealing is used",
    )
    vae_group.add_argument(
        "--beta_step",
        type=float,
        default=1.1,
        help="multiplicative step size for beta, if using beta annealing",
    )
    vae_group.add_argument(
        "--beta_reset_every",
        type=int,
        default=1e10,
        help='Reset beta (cyclic scheduling)'
    )
    vae_group.add_argument(
        "--beta_step_freq",
        type=int,
        default=3,
        help="frequency for beta step, if taking a step for beta",
    )
    vae_group.add_argument(
        "--beta_warmup",
        type=int,
        default=10,
        help="number of iterations of warmup before beta starts increasing",
    )
    vae_group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    vae_group.add_argument(
        "--use_pretrained",
        action='store_true',
        help="Use trained VAE",
    )

    args = parser.parse_args()

    if not is_robust(args.acq_func_id):
        args.estimate_rec_error = 0
    if 'ErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
        assert 'eps' in args.acq_func_kwargs
    elif 'MultiObjectiveErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs

    args.dataset_path = os.path.join(ROOT_PROJECT,
                                     get_filepath(args.ignore_percentile, args.dataset_path, args.data_seed,
                                                  good_percentile=args.good_percentile))
    if args.pretrained_model_file is not None:
        args.pretrained_model_file = os.path.join(get_storage_root(), args.pretrained_model_file)
    elif args.semi_supervised and args.use_pretrained:

        args.pretrained_model_file = get_pretrained_model_path(
            version=args.version,
            k="inf",
            ignore_percentile=args.ignore_percentile,
            good_percentile=args.good_percentile,
            n_max_epochs=args.training_max_epochs,
            predict_target=False,
            metric_loss=None,
            metric_loss_kw={},
            latent_dim=args.latent_dim,
            hdims=args.target_predictor_hdims,
            beta_final=args.beta_final,
            beta_metric_loss=args.beta_metric_loss,
            beta_target_pred_loss=args.beta_target_pred_loss,
        )
    elif args.use_pretrained:
        args.pretrained_model_file = get_pretrained_model_path(
            version=args.version,
            k=args.rank_weight_k,
            ignore_percentile=args.ignore_percentile,
            good_percentile=args.good_percentile,
            n_max_epochs=args.training_max_epochs,
            predict_target=args.predict_target,
            metric_loss=args.metric_loss,
            metric_loss_kw=args.metric_loss_kw,
            latent_dim=args.latent_dim,
            hdims=args.target_predictor_hdims,
            beta_final=args.beta_final,
            beta_metric_loss=args.beta_metric_loss,
            beta_target_pred_loss=args.beta_target_pred_loss,
        )
    # Seeding
    pl.seed_everything(args.seed)

    # create result directory
    result_dir = get_path(
        lso_strategy=args.lso_strategy,
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        r=args.retraining_frequency,
        ignore_percentile=args.ignore_percentile,
        good_percentile=args.good_percentile,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        metric_loss=args.metric_loss,
        metric_loss_kw=args.metric_loss_kw,
        beta_final=args.beta_final,
        beta_metric_loss=args.beta_metric_loss,
        beta_target_pred_loss=args.beta_target_pred_loss,
        acq_func_id=args.acq_func_id,
        acq_func_kwargs=args.acq_func_kwargs,
        covar_name=args.covar_name,
        input_wp=args.input_wp,
        seed=args.seed,
        random_search_type=args.random_search_type,
        n_max_epochs=args.training_max_epochs,
        estimate_rec_error=args.estimate_rec_error,
        cost_aware_gamma_sched=args.cost_aware_gamma_sched,
        semi_supervised=args.semi_supervised,
        use_pretrained=args.use_pretrained
    )
    print(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    save_w_pickle(args, result_dir, 'args.pkl')
    logs = ''
    exc: Optional[Exception] = None
    try:
        main_aux(args, result_dir=result_dir)
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc


def main_aux(args, result_dir):
    """ main """

    device = args.cuda
    if device is not None:
        torch.cuda.set_device(device)
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    # get initial dataset
    datamodule = WeightedExprDataset(args, DataWeighter(args))
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)
    # data_str, data_enc, data_scores = expr_data.get_initial_dataset_and_weights(
    #     Path(args.data_dir), args.ignore_percentile, args.n_data)

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # Load model
    data_info = grammar.gram.split('\n')
    if args.use_pretrained:
        print(f'Use pretrained VAE from: {args.pretrained_model_file}')
        vae: EquationVaeTorch = EquationVaeTorch.load_from_checkpoint(args.pretrained_model_file,
                                                                      charset_length=len(data_info),
                                                                      max_length=MAX_LEN)
        vae.hparams.cuda = args.cuda
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.metric_loss = args.metric_loss
        vae.hparams.metric_loss = args.metric_loss
        vae.beta_metric_loss = args.beta_metric_loss
        vae.hparams.beta_metric_loss = args.beta_metric_loss
        vae.metric_loss_kw = args.metric_loss_kw
        vae.hparams.metric_loss_kw = args.metric_loss_kw
        vae.predict_target = args.predict_target
        vae.hparams.predict_target = args.predict_target
        vae.beta_target_pred_loss = args.beta_target_pred_loss
        vae.hparams.beta_target_pred_loss = args.beta_target_pred_loss
        if vae.predict_target and vae.target_predictor is None:
            vae.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
            vae.build_target_predictor()
    else:
        print('Train VAE from scratch')
        vae: EquationVaeTorch = EquationVaeTorch(args, charset_length=len(data_info), max_length=MAX_LEN)
    vae.eval()

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=float(datamodule.prop_train.min()),
        n_train=len(datamodule.train_dataset),
        save_path=result_dir
    )

    start_num_retrain = 0

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_point_errors=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
        rand_point_due_bo_fail=[],  # binary entry: 0 -> bo worked, 1 -> bo failed so sampled a point at random
    )

    result_filepath = os.path.join(result_dir, 'results.npz')
    if not args.overwrite and os.path.exists(result_filepath):
        with np.load(result_filepath, allow_pickle=True) as npz:
            results = {}
            for k in list(npz.keys()):
                results[k] = npz[k]
                if k != 'params':
                    results[k] = list(results[k])
                else:
                    results[k] = npz[k].item()
            if 'rand_point_due_bo_fail' not in results:
                results['rand_point_due_bo_fail'] = [0] * len(results['opt_points'])
        start_num_retrain = results['opt_model_version'][-1] + 1

        prev_retrain_model = args.retraining_frequency * (start_num_retrain - 1)
        num_sampled_points = len(results['opt_points'])

        if args.n_init_retrain_epochs == 0 and prev_retrain_model == 0:
            pretrained_model_path = args.pretrained_model_path
        else:
            pretrained_model_path = os.path.join(result_dir, 'retraining', f'retrain_{prev_retrain_model}',
                                                 'checkpoints', 'last.ckpt')
        ckpt = torch.load(pretrained_model_path, map_location=f"cuda:{args.cuda}")
        ckpt['hyper_parameters']['hparams'].metric_loss = args.metric_loss
        ckpt['hyper_parameters']['hparams'].metric_loss_kw = args.metric_loss_kw
        ckpt['hyper_parameters']['hparams'].beta_metric_loss = args.beta_metric_loss
        ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
        if args.predict_target:
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
        torch.save(ckpt, pretrained_model_path)
        vae.load_from_checkpoint(pretrained_model_path, charset_length=len(data_info), max_length=MAX_LEN)
        if args.predict_target and not hasattr(vae.hparams, "predict_target"):
            vae.hparams.predict_target = args.predict_target
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
        vae.hparams.cuda = args.cuda
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.eval()

        # Set up some stuff for the progress bar
        num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency)) - start_num_retrain

        model: EquationGrammarModelTorch = EquationGrammarModelTorch(vae)

        datamodule.append_train_data(np.array([model.smiles_to_one_hot([x])[0] for x in results['opt_points']]),
                                     np.array(results['opt_point_properties']), np.array(results['opt_points']))
        postfix = dict(
            retrain_left=num_retrain,
            best=float(datamodule.prop_train.min()),
            n_train=len(datamodule.train_dataset),
            initial=num_sampled_points,
            save_path=result_dir
        )
        print(f"Retrain from {result_dir} | Best: {min(results['opt_point_properties'])}")
    start_time = time.time()

    # Main loop
    with tqdm(
            total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(start_num_retrain, start_num_retrain + num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")
            print(result_dir)
            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = os.path.join(result_dir, "retraining")
                version = f"retrain_{samples_so_far}"
                retrain_model(
                    model=vae, datamodule=datamodule, save_dir=retrain_dir,
                    version_str=version, num_epochs=num_epochs, cuda=args.cuda,
                    semi_supervised=args.semi_supervised,
                )
                vae.eval()
            del num_epochs

            model: EquationGrammarModelTorch = EquationGrammarModelTorch(vae)

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Draw samples for logs!
            if args.samples_per_model > 0:
                sample_x, sample_y = latent_sampling(
                    model=model, n_decode=args.n_decode_attempts,
                    num_queries_to_do=args.samples_per_model
                )

                # Append to results dict
                results["sample_points"].append(sample_x)
                results["sample_properties"].append(sample_y)
                results["sample_versions"].append(ret_idx)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            error_new = None
            if args.lso_strategy == "opt":
                gp_dir = os.path.join(result_dir, "gp", f"iter{samples_so_far}")
                os.makedirs(gp_dir, exist_ok=True)
                gp_data_file = os.path.join(gp_dir, "data.npz")
                x_new, y_new, rand_point_due_bo_fail, error_new = latent_optimization(
                    model=model,
                    datamodule=datamodule,
                    n_inducing_points=args.n_inducing_points,
                    n_best_points=args.n_best_points,
                    n_rand_points=args.n_rand_points,
                    tkwargs=tkwargs,
                    num_queries_to_do=num_queries_to_do,
                    use_test_set=args.use_test_set,
                    use_full_data_for_gp=args.use_full_data_for_gp,
                    gp_data_file=gp_data_file,
                    gp_run_folder=gp_dir,
                    test_gp_error_fit=args.test_gp_error_fit,
                    n_decode_attempts=args.n_decode_attempts,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    acq_func_id=args.acq_func_id,
                    acq_func_kwargs=args.acq_func_kwargs,
                    acq_func_opt_kwargs=args.acq_func_opt_kwargs,
                    q=1,
                    num_restarts=args.num_restarts,
                    raw_initial_samples=args.raw_initial_samples,
                    num_MC_sample_acq=args.num_MC_sample_acq,
                    input_wp=args.input_wp,
                    estimate_rec_error=args.estimate_rec_error,
                    cost_aware_gamma_sched=args.cost_aware_gamma_sched,
                    pbar=pbar,
                    postfix=postfix,
                    semi_supervised=args.semi_supervised,
                )
            elif args.lso_strategy == "sample":
                x_new, y_new = latent_sampling(
                    model=model, n_decode=args.n_decode_attempts, num_queries_to_do=num_queries_to_do,
                )
                rand_point_due_bo_fail = [0] * num_queries_to_do
            elif args.lso_strategy == "random_search":
                x_new, y_new = latent_random_search(
                    model=model, n_decode=args.n_decode_attempts, num_queries_to_do=num_queries_to_do, tkwargs=tkwargs,
                    datamodule=datamodule, filter_unique=False,
                    random_search_type=args.random_search_type, seed=args.seed, fast_forward=samples_so_far
                )
                rand_point_due_bo_fail = [0] * num_queries_to_do
            else:
                raise NotImplementedError(args.lso_strategy)

            # Update dataset
            datamodule.append_train_data(
                np.array([model.smiles_to_one_hot([x])[0] for x in x_new if x is not None]),
                y_new, x_new)

            # Add new results
            results["opt_points"] += list(x_new)
            results["opt_point_properties"] += list(y_new)
            if error_new is not None:
                results['opt_point_errors'] += list(error_new)
            results["opt_model_version"] += [ret_idx] * len(x_new)
            results["rand_point_due_bo_fail"] += rand_point_due_bo_fail

            postfix["best"] = min(postfix["best"], float(min(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset)
            pbar.set_postfix(postfix)

            # Keep a record of the dataset here
            new_data_file = os.path.join(result_dir, f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
                                         )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.expr_train))

            # Save results
            np.savez_compressed(os.path.join(result_dir, "results.npz"), **results)

    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


def latent_optimization(
        model: EquationGrammarModelTorch,
        datamodule: WeightedExprDataset,
        n_inducing_points: int,
        n_best_points: int,
        n_rand_points: int,
        tkwargs: Dict[str, Any],
        num_queries_to_do: int,
        use_test_set: bool,
        use_full_data_for_gp: bool,
        gp_data_file: str,
        gp_run_folder: str,
        test_gp_error_fit: bool,
        n_decode_attempts: int,
        scale: bool,
        covar_name: str,
        acq_func_id: str,
        acq_func_kwargs: Dict[str, Any],
        acq_func_opt_kwargs: Dict[str, Any],
        q: int,
        num_restarts: int,
        raw_initial_samples: int,
        num_MC_sample_acq: int,
        input_wp: bool,
        estimate_rec_error: bool,
        cost_aware_gamma_sched: Optional[str],
        pbar=None,
        postfix=None,
        semi_supervised: bool = False,
):
    ##################################################
    # Prepare GP
    ##################################################

    # First, choose GP points to train!
    model.vae.to(**tkwargs)
    X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std, train_inds, test_inds = expr_data.get_latent_encodings(
        use_test_set=use_test_set,
        use_full_data_for_gp=use_full_data_for_gp,
        model=model,
        data_file=gp_data_file,
        data_scores=datamodule.prop_train,
        data_str=datamodule.expr_train,
        n_best=n_best_points,
        n_rand=n_rand_points,
        tkwargs=tkwargs,
        return_inds=True
    )

    # do not standardize -> we'll normalize in unit cube
    X_train = torch.tensor(X_train).to(**tkwargs)

    # standardise targets
    y_train = torch.tensor(y_train).to(**tkwargs)

    do_robust = is_robust(acq_func_id)
    error_train: Optional[Tensor] = None
    if do_robust:
        # get reconstruction error on X_train
        error_train = get_rec_x_error(model, tkwargs=tkwargs,
                                      one_hots=torch.from_numpy(datamodule.data_train[train_inds]),
                                      zs=X_train)

        assert error_train.shape == y_train.shape == (len(X_train), 1), (error_train.shape, y_train.shape)

    model.vae.cpu()  # Make sure to free up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    curr_gp_file = None
    curr_gp_error_file = None

    all_new_exprs = []
    all_new_scores = []
    all_new_errors = [] if do_robust else None
    all_z_opt_noise = []
    all_expr_noise = []
    all_score_noise = []

    n_rand_acq = 0  # number of times we have to acquire a random point as bo acquisition crashed

    rand_point_due_bo_fail = []
    # for gp_iter in range(num_queries_to_do):
    gp_iter = 0
    redo_counter = 1
    while len(all_new_exprs) < num_queries_to_do:
        # Part 1: fit GP
        # ===============================
        new_gp_file = os.path.join(gp_run_folder, f"gp_train_res{gp_iter:04d}.npz")
        new_gp_error_file = os.path.join(gp_run_folder, f"gp_train_error_res{gp_iter:04d}.npz")
        # log_path = os.path.join(gp_run_folder, f"gp_train{gp_iter:04d}.log")
        iter_seed = int(np.random.randint(10000))

        gp_file = None
        gp_error_file = None
        if gp_iter == 0:
            # Add commands for initial fitting
            gp_fit_desc = "GP initial fit"
            # n_perf_measure = 0
            current_n_inducing_points = min(X_train.shape[0], n_inducing_points)
        else:
            gp_fit_desc = "GP incremental fit"
            gp_file = curr_gp_file
            gp_error_file = curr_gp_error_file
            # n_perf_measure = 1  # specifically see how well it fits the last point!
        init = gp_iter == 0
        # if semi-supervised training, wait until we have enough points to use as many inducing points as
        # we wanted and re-init GP
        if X_train.shape[0] == n_inducing_points:
            current_n_inducing_points = n_inducing_points
            init = True

        old_desc = None
        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        np.random.seed(iter_seed)

        # To account for outliers
        bounds = torch.zeros(2, X_train.shape[1], **tkwargs)
        bounds[0] = torch.quantile(X_train, .0005, dim=0)
        bounds[1] = torch.quantile(X_train, .9995, dim=0)
        ybounds = torch.zeros(2, y_train.shape[1], **tkwargs)
        ybounds[0] = torch.quantile(-y_train, .0005, dim=0)
        ybounds[1] = torch.quantile(-y_train, .9995, dim=0)
        ydelta = .05 * (ybounds[1] - ybounds[0])
        ybounds[0] -= ydelta
        ybounds[1] += ydelta
        # make sure best sample is within bounds
        y_train_std = y_train.add(-y_train.mean()).div(y_train.std())
        y_train_normalized = normalize(-y_train, ybounds)  # minimize
        bounds = put_max_in_bounds(X_train, -y_train_std, bounds)
        # bounds = put_max_in_bounds(X_train, y_train_normalized, bounds)

        # print(f"Data bound of {bounds} found...")
        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta
        # print(f"Using data bound of {bounds}...")

        train_x = normalize(X_train, bounds)
        try:
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=-y_train_std,  # minimize
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=init,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 5000} if init else {'lr': 1e-4, 'maxiter': 100}
            )
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting objective GP from scratch...")
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=-y_train_std,  # minimize
                # train_y=y_train_normalized,
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=True,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 5000}
            )
        curr_gp_file = new_gp_file

        # create bounds on posterior variance to use in acqf scheduling
        with torch.no_grad():
            y_pred_var = gp_model.posterior(train_x).variance
            yvarbounds = torch.zeros(2, y_train.shape[1], **tkwargs)
            yvarbounds[0] = torch.quantile(y_pred_var, .0005, dim=0)
            yvarbounds[1] = torch.quantile(y_pred_var, .9995, dim=0)
            yvardelta = .05 * (yvarbounds[1] - yvarbounds[0])
            yvarbounds[0] -= yvardelta
            yvarbounds[1] += yvardelta

        if do_robust:
            if estimate_rec_error or init:
                # (re)train model only at initialisation or if new error values have been added
                rbounds = torch.zeros(2, error_train.shape[1], **tkwargs)
                rbounds[0] = torch.quantile(error_train, .0005, dim=0)
                rbounds[1] = torch.quantile(error_train, .9995, dim=0)
                rdelta = .05 * (rbounds[1] - rbounds[0])
                rbounds[0] -= rdelta
                rbounds[1] += rdelta
                error_train_normalized = normalize(error_train, rbounds)
                error_train_std = error_train.add(-error_train.mean()).div(error_train.std())
                # error_train_dstd = error_train.div(error_train.std())
                # error_train_sigmoid = torch.sigmoid(error_train)
                # error_train_sigmoid_inverse = torch.log((1-error_train).div(error_train))
                try:
                    gp_model_error = gp_torch_train(
                        train_x=train_x,
                        train_y=error_train,
                        n_inducing_points=current_n_inducing_points,
                        tkwargs=tkwargs,
                        init=init,
                        scale=scale,
                        covar_name=covar_name,
                        gp_file=gp_error_file,
                        save_file=new_gp_error_file,
                        input_wp=input_wp,
                        outcome_transform=Standardize(m=1),
                        options={'lr': 5e-3, 'maxiter': 5000} if init else {'lr': 1e-4, 'maxiter': 100}
                    )
                except (RuntimeError, NotPSDError) as e:  # Random acquisition
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting error GP from scratch...")
                    gp_model_error = gp_torch_train(
                        train_x=train_x,
                        train_y=error_train,
                        n_inducing_points=current_n_inducing_points,
                        tkwargs=tkwargs,
                        init=True,
                        scale=scale,
                        covar_name=covar_name,
                        gp_file=gp_error_file,
                        save_file=new_gp_error_file,
                        input_wp=input_wp,
                        outcome_transform=Standardize(m=1),
                        options={'lr': 5e-3, 'maxiter': 5000}
                    )
                curr_gp_error_file = new_gp_error_file

                # create bounds on posterior variance to use in acqf scheduling
                with torch.no_grad():
                    r_pred_var = gp_model_error.posterior(train_x).variance
                    rvarbounds = torch.zeros(2, error_train.shape[1], **tkwargs)
                    rvarbounds[0] = torch.quantile(r_pred_var, .0005, dim=0)
                    rvarbounds[1] = torch.quantile(r_pred_var, .9995, dim=0)
                    rvardelta = .05 * (rvarbounds[1] - rvarbounds[0])
                    rvarbounds[0] -= rvardelta
                    rvarbounds[1] += rvardelta
        else:
            gp_model_error = None

        if test_gp_error_fit:
            # GP test on validation set
            gp_test_folder = os.path.join(gp_run_folder, f"gp_validation_res{gp_iter:04d}")
            with torch.no_grad():
                model.eval()
                model.vae.to(**tkwargs)
                x_orig_val = torch.from_numpy(datamodule.data_val).to(**tkwargs)
                X_val = model.vae.encode_to_params(x_orig_val)[0].to(**tkwargs)
                x_val = normalize(X_val, bounds)
                y_val = torch.from_numpy(datamodule.prop_val).to(**tkwargs)
                y_val_std = y_val.add(-y_train.mean()).div(y_train.std())
                y_val_normalized = normalize(y_val, ybounds)

            # create test points
            error_val: Optional[Tensor] = None
            with torch.no_grad():
                if do_robust:
                    error_val = get_rec_x_error(model=model, tkwargs=tkwargs, zs=X_val, one_hots=x_orig_val)
                    if rbounds is not None:
                        error_val_normalized = normalize(error_val, rbounds)
                    error_val_std = error_val.add(-error_train.mean()).div(error_train.std())

            gp_fit_test(x_train=train_x, y_train=-y_train_std, error_train=error_train,
                        x_test=x_val, y_test=-y_val_std, error_test=error_val,
                        gp_obj_model=gp_model, gp_error_model=gp_model_error,
                        tkwargs=tkwargs, gp_test_folder=gp_test_folder,
                        )

            # --acq-func-id=ErrorAwareEI
            # --acq-func-kwargs={'gamma':1,'eps':10,'configuration':'ratio'}
            # --cost_aware_gamma_sched=post_var_inv_tradeoff
            # --metric_loss=triplet
            # --metric_loss_kw={'threshold':.1,'soft':1}

            # make_local_stationarity_plots(centers=train_x[np.random.choice(np.arange(3000), 1000)],
            #                               radiuses=np.arange(1, 101) / 20, n_samples=100, model=model,
            #                               score_function=expr_data.score_function, target=None, dist='l1',
            #                               save_dir=gp_test_folder)

            # save data
            name = f"z-{model.vae.hparams.get('latent_dim')}"
            name += "_" + model.vae.hparams.get('metric_loss', None) if model.vae.hparams.get('metric_loss',
                                                                                              None) is not None else ""
            name += "-soft" if model.vae.hparams.get('metric_loss_kw') is not None and model.vae.hparams.get(
                'metric_loss_kw').get('soft') is not None else ""
            name += "_predy" if model.vae.hparams['predict_target'] else ""
            name += "_inp_wp" if input_wp else ""
            name += "_semi_supervised" if semi_supervised else ""
            test_data_folder = os.path.join(get_storage_root(), 'logs/test/expr/data', name)
            if not os.path.exists(test_data_folder):
                os.makedirs(test_data_folder)

            torch.save(train_x, os.path.join(test_data_folder, f"x_train.pt"))
            torch.save(x_val, os.path.join(test_data_folder, f"x_val.pt"))
            torch.save(y_train_normalized, os.path.join(test_data_folder, f"y_train_normalized.pt"))
            torch.save(y_train_std, os.path.join(test_data_folder, f"y_train_std.pt"))
            torch.save(y_train, os.path.join(test_data_folder, f"y_train.pt"))
            torch.save(y_val_normalized, os.path.join(test_data_folder, f"y_val_normalized.pt"))
            torch.save(y_val_std, os.path.join(test_data_folder, f"y_val_std.pt"))
            torch.save(y_val, os.path.join(test_data_folder, f"y_val.pt"))

            if do_robust:
                torch.save(error_train_normalized, os.path.join(test_data_folder, f"r_train_normalized.pt"))
                torch.save(error_train_std, os.path.join(test_data_folder, f"r_train_std.pt"))
                torch.save(error_train, os.path.join(test_data_folder, f"r_train.pt"))
                torch.save(error_val_normalized, os.path.join(test_data_folder, f"r_val_normalized.pt"))
                torch.save(error_val_std, os.path.join(test_data_folder, f"r_val_std.pt"))
                torch.save(error_val, os.path.join(test_data_folder, f"r_val.pt"))

            from weighted_retraining.test.test_retraining import make_latent_space_distance_plots
            make_latent_space_distance_plots(
                x_train=train_x,
                y_train=y_train_std,
                x_val=x_val,
                y_val=y_val_std,
                plot_folder=gp_test_folder,
                dist='l2',
                name=name,
            )
            make_latent_space_distance_plots(
                x_train=train_x,
                y_train=y_train_std,
                x_val=x_val,
                y_val=y_val_std,
                plot_folder=gp_test_folder,
                dist='l1',
                name=name
            )
            make_latent_space_distance_plots(
                x_train=train_x,
                y_train=y_train_std,
                x_val=x_val,
                y_val=y_val_std,
                plot_folder=gp_test_folder,
                dist='cos',
                name=name
            )

            print("Done testing GPs.")

        # Part 2: optimize GP acquisition func to query point
        # ===============================

        # Run GP opt script
        # opt_path = os.path.join(gp_run_folder, f"gp_opt_res{gp_iter:04d}.npy")
        # log_path = os.path.join(gp_run_folder, f"gp_opt_{gp_iter:04d}.log")

        if pbar is not None:
            pbar.set_description("optimizing acq func")

        print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via optimization...")
        try:  # BO acquisition
            if do_robust:
                if cost_aware_gamma_sched is not None:
                    # assert isinstance(acq_func_kwargs['gamma'], float) or isinstance(acq_func_kwargs['gamma'], int), acq_func_kwargs['gamma']
                    if 'gamma_start' not in acq_func_kwargs:
                        acq_func_kwargs['gamma_start'] = float(acq_func_kwargs['gamma'])
                    if cost_aware_gamma_sched == 'linear':
                        acq_func_kwargs['gamma'] = (num_queries_to_do - len(all_new_scores)) / num_queries_to_do * \
                                                   acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'reverse_linear':
                        acq_func_kwargs['gamma'] = len(all_new_scores) / num_queries_to_do * acq_func_kwargs[
                            'gamma_start']
                    elif cost_aware_gamma_sched == 'exponential':
                        acq_func_kwargs['gamma'] = 0.75 ** len(all_new_scores) * acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'reverse_exponential':
                        acq_func_kwargs['gamma'] = 0.75 ** (num_queries_to_do - len(all_new_scores)) * acq_func_kwargs[
                            'gamma_start']
                    elif cost_aware_gamma_sched == 'fixed':
                        acq_func_kwargs['gamma'] = acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'post_obj_var' \
                            or cost_aware_gamma_sched == 'post_obj_inv_var' \
                            or cost_aware_gamma_sched == 'post_err_var' \
                            or cost_aware_gamma_sched == 'post_err_inv_var' \
                            or cost_aware_gamma_sched == 'post_min_var' \
                            or cost_aware_gamma_sched == 'post_var_tradeoff' \
                            or cost_aware_gamma_sched == 'post_var_inv_tradeoff':
                        acq_func_kwargs.update({'gamma': cost_aware_gamma_sched})
                    else:
                        raise ValueError(cost_aware_gamma_sched)

                acq_func_kwargs['y_var_bounds'] = yvarbounds
                acq_func_kwargs['r_var_bounds'] = rvarbounds
                print(acq_func_kwargs)

                res = bo_mo_loop(
                    gp_model=gp_model,
                    gp_model_error=gp_model_error,
                    vae_model=model,
                    acq_func_id=acq_func_id,
                    acq_func_kwargs=acq_func_kwargs,
                    acq_func_opt_kwargs=acq_func_opt_kwargs,
                    bounds=normalize(bounds, bounds),
                    tkwargs=tkwargs,
                    q=q,
                    num_restarts=num_restarts,
                    raw_initial_samples=raw_initial_samples,
                    seed=iter_seed,
                    num_MC_sample_acq=num_MC_sample_acq,
                )
                model.vae.eval()
                z_opt = res
                gp_model_error.cpu()
            else:
                print('robust_opt_expr', acq_func_id)
                res = bo_loop(
                    gp_model=gp_model,
                    acq_func_id=acq_func_id,
                    acq_func_kwargs=acq_func_kwargs,
                    acq_func_opt_kwargs=acq_func_opt_kwargs,
                    bounds=normalize(bounds, bounds),
                    tkwargs=tkwargs,
                    q=q,
                    num_restarts=num_restarts,
                    raw_initial_samples=raw_initial_samples,
                    seed=iter_seed,
                    num_MC_sample_acq=num_MC_sample_acq,
                )
                z_opt = res
            rand_point_due_bo_fail += [0] * q
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via random sampling...")
            n_rand_acq += q
            z_opt = torch.rand(q, bounds.shape[1]).to(bounds)
            exc = e
            rand_point_due_bo_fail += [1] * q
        gp_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        z_opt = unnormalize(z_opt, bounds).cpu().detach()
        z_opt = torch.atleast_2d(z_opt)

        assert q == 1, q
        # compute and save new inputs and corresponding scores
        new_expr = model.decode_from_latent_space(zs=z_opt, n_decode_attempts=n_decode_attempts)
        new_score = expr_data.score_function(new_expr)
        print('Got new Expression', new_expr)
        # if new expression is decoded to None it is invalid so redo step altogether
        if new_expr is None or new_expr.item() is None:
            if redo_counter == 3:
                print(f'Invalid! z_opt decoded to None in {n_decode_attempts} attempts at iteration {gp_iter} '
                      f'even after {redo_counter} restarts -> moving on with a randomly picked point...')
                redo_counter = 1
                while new_expr is None or new_expr.item() is None:
                    z_opt = torch.rand(q, bounds.shape[1]).to(bounds)
                    new_expr = model.decode_from_latent_space(zs=z_opt, n_decode_attempts=n_decode_attempts)
                new_score = expr_data.score_function(new_expr)
                rand_point_due_bo_fail += [1] * q
            else:
                print(f'Invalid! z_opt decoded to None in {n_decode_attempts} attempts -> Re-doing iteration {gp_iter}')
                redo_counter += 1
                continue
        # if expr is already in training set perturb z_opt & decode until a new expr is found or restart BO step
        all_new_z_opt_noise = []
        all_new_expr_noise = []
        all_new_score_noise = []
        if new_expr in datamodule.expr_train or new_expr in all_expr_noise:
            print(f"Expression {new_expr[0]} is already in training set -> perturbing z_opt 10 times...")
            all_new_z_opt_noise.append(z_opt)
            all_new_expr_noise.append(new_expr)
            all_new_score_noise.append(new_score)
            noise_level = 1e-2
            for random_trials in range(10):
                z_opt_noise = z_opt + torch.randn_like(z_opt) * noise_level
                new_expr_noise = model.decode_from_latent_space(zs=z_opt_noise, n_decode_attempts=n_decode_attempts)
                if new_expr_noise is None or new_expr_noise.item() is None:
                    print(f"... skipping perturbed point decoded to {new_expr_noise} ...")
                else:
                    new_score_noise = expr_data.score_function(new_expr_noise)
                    all_new_z_opt_noise.append(z_opt_noise)
                    all_new_expr_noise.append(new_expr_noise)
                    all_new_score_noise.append(new_score_noise)
                    if new_expr_noise in datamodule.expr_train or new_expr in all_expr_noise:
                        # print(f"... {new_expr_noise} already in training set with score {new_score_noise}...")
                        noise_level *= 1.1
                    else:
                        z_opt = z_opt_noise
                        new_expr = new_expr_noise
                        new_score = expr_data.score_function(new_expr)
                        print(f'...after {random_trials} perturbations got {new_expr} with score {new_score}')
                        break
            if random_trials == 9:
                if redo_counter == 3:
                    print(f"Moving on anyway after redoing BO step {gp_iter} 10 times.")
                else:
                    print(f"...did not find any new expression after perturbing for {random_trials} times "
                          f"-> Re-doing BO step {gp_iter} altogether.")
                    redo_counter += 1
                    continue

        all_new_z_opt_noise = [z_opt] if all_new_z_opt_noise == [] else all_new_z_opt_noise
        all_new_expr_noise = [new_expr] if all_new_expr_noise == [] else all_new_expr_noise
        all_new_score_noise = [new_score] if all_new_score_noise == [] else all_new_score_noise
        all_new_z_opt_noise = torch.cat(all_new_z_opt_noise).to(**tkwargs)
        all_new_expr_noise = np.array(all_new_expr_noise).flatten()
        all_new_score_noise = np.array(all_new_score_noise).flatten()
        all_z_opt_noise = torch.cat([torch.tensor(all_z_opt_noise).to(**tkwargs), all_new_z_opt_noise]).to(**tkwargs)
        all_expr_noise = np.concatenate([all_expr_noise, all_new_expr_noise])
        all_score_noise = np.concatenate([all_score_noise, all_new_score_noise])

        all_new_exprs = np.append(all_new_exprs, new_expr)
        all_new_scores = np.append(all_new_scores, new_score)
        print_flush(f"\t\tPicked new input: {all_new_exprs[-1]} with value {all_new_scores[-1]}...")

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)
            pbar.update(len(z_opt))

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = min(postfix["best"], float(min(all_new_scores)))
                pbar.set_postfix(postfix)

        if do_robust and estimate_rec_error:
            # add estimate new errors
            new_errors = expr_data.get_rec_error_emb(
                model=model,
                tkwargs=tkwargs,
                exprs=all_new_expr_noise
            ).cpu().numpy()
            all_new_errors = np.append(all_new_errors, new_errors[-q:])

            new_errors = torch.from_numpy(new_errors).reshape(-1, 1)
        else:
            new_errors = None

        aux_res_datasets = expr_data.append_trainset_torch(X_train, y_train,
                                                           new_inputs=all_new_z_opt_noise,
                                                           new_scores=torch.from_numpy(all_new_score_noise).reshape(-1,
                                                                                                                    1),
                                                           y_errors=error_train,
                                                           new_errors=new_errors)
        if new_errors is None:
            X_train, y_train = aux_res_datasets
        else:
            X_train, y_train, error_train = aux_res_datasets

        gp_iter += 1
        redo_counter = 1

    if n_rand_acq / num_queries_to_do > .2:
        raise ValueError(
            f'Sampled too many random points ({n_rand_acq} / {num_queries_to_do}) due to exceptions in BO such as {exc}')
    elif n_rand_acq > 0:
        print(f'Acquired {n_rand_acq} / {num_queries_to_do} points at random due to bo acquisition failure {exc}')
    return all_new_exprs, all_new_scores, rand_point_due_bo_fail, all_new_errors


def latent_sampling(model: EquationGrammarModelTorch, n_decode: int, num_queries_to_do: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """ Draws samples from latent space and appends to the dataset """

    print_flush("\t\tPicking new inputs via sampling...")
    new_latents = np.random.randn(num_queries_to_do, model.vae.latent_dim)
    new_inputs = model.decode_from_latent_space(zs=new_latents, n_decode_attempts=n_decode)
    new_scores = expr_data.score_function(new_inputs)

    return new_inputs, new_scores


def latent_random_search(model: EquationGrammarModelTorch, n_decode: int, num_queries_to_do: int,
                         tkwargs: Dict[str, Any], datamodule: WeightedExprDataset, filter_unique=False,
                         random_search_type=None,
                         seed: int = None, fast_forward: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws samples from search space obtained through encoding of inputs in the dataset

    Args:
        model: generative model for LSO
        n_decode: number of decoding attempts to get a valid equation
        num_queries_to_do: number of queries
        tkwargs: dtype and device for torch tensors and models
        datamodule: equation expression dataset
    """
    print_flush(f"\t\tPicking new inputs via {random_search_type if random_search_type is not None else ''} RS...")

    # get_latent_encodings
    # Add budget to the filename
    model.vae.to(**tkwargs)
    X_enc = torch.tensor(get_latent_encodings_aux(
        model=model,
        data_str=datamodule.expr_train
    ))

    model.vae.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()

    if random_search_type == 'sobol':
        assert seed is not None, 'Should specify seed for sobol random search'
        soboleng = torch.quasirandom.SobolEngine(dimension=X_enc.shape[1], scramble=True, seed=seed)
        soboleng.fast_forward(fast_forward)

    new_inputs_ = []
    new_scores_ = []

    while len(new_inputs_) < num_queries_to_do:
        # To account for outliers
        bounds = torch.zeros(2, X_enc.shape[1]).to(X_enc)
        bounds[0] = torch.quantile(X_enc, .0005, dim=0)
        bounds[1] = torch.quantile(X_enc, .9995, dim=0)
        # make sure best sample is within bounds
        bounds = put_max_in_bounds(X_enc, -torch.tensor(datamodule.prop_train).unsqueeze(-1).to(X_enc),
                                   bounds)

        # print(f"Data bound of {bounds} found...")
        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta

        if random_search_type is None:
            bounds = bounds.detach().numpy()
            new_latents = np.random.rand(1, model.vae.latent_dim) * (bounds[1] - bounds[0]) + bounds[0]
        elif random_search_type == 'sobol':
            new_latents = unnormalize(soboleng.draw(1).to(bounds), bounds).cpu().numpy()
        else:
            raise ValueError(f'{random_search_type} not supported for random search')

        with torch.no_grad():
            new_inputs = model.decode_from_latent_space(zs=new_latents, n_decode_attempts=n_decode)

        # redo iteration if expression is None or not novel
        if None in new_inputs or new_inputs in datamodule.expr_train or new_inputs in new_inputs_:
            continue

        new_scores = expr_data.score_function(new_inputs)

        new_inputs_ = np.append(new_inputs_, new_inputs)
        new_scores_ = np.concatenate([new_scores_, new_scores])
        X_enc = torch.cat([X_enc, torch.from_numpy(new_latents).reshape(-1, model.vae.latent_dim)], 0)

    model.vae.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()
    return new_inputs_[:num_queries_to_do], new_scores_[:num_queries_to_do]


if __name__ == "__main__":
    main()
