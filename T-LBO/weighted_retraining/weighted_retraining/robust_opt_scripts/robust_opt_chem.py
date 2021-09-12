import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm, trange

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
if os.path.join(ROOT_PROJECT, 'weighted_retraining') in sys.path:
    sys.path.remove(os.path.join(ROOT_PROJECT, 'weighted_retraining'))
sys.path[0] = ROOT_PROJECT

from weighted_retraining.weighted_retraining.utils import SubmissivePlProgressbar, DataWeighter, print_flush

from utils.utils_cmd import parse_list, parse_dict
from utils.utils_save import get_storage_root, save_w_pickle, str_dict
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from weighted_retraining.weighted_retraining.robust_opt_scripts.base import add_common_args

# My imports
from weighted_retraining.weighted_retraining import GP_TRAIN_FILE, GP_OPT_FILE
from weighted_retraining.weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset,
    WeightedMolTreeFolder,
    get_rec_x_error)
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet
from weighted_retraining.weighted_retraining.robust_opt_scripts import base as wr_base

logger = logging.getLogger("chem-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def _run_command(command, command_name):
    logger.debug(f"{command_name} command:")
    logger.debug(command)
    start_time = time.time()
    run_result = subprocess.run(command, capture_output=True)
    assert run_result.returncode == 0, run_result.stderr
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def _batch_decode_z_and_props(
        model: JTVAE,
        z: torch.Tensor,
        datamodule: WeightedJTNNDataset,
        invalid_score: float,
        pbar: tqdm = None,
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j: j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Calculate objective function values and choose which points to keep
    # Invalid points get a value of None
    z_prop = [
        invalid_score if s is None else datamodule.train_dataset.prop_func(s)
        for s in z_decode
    ]

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def _choose_best_rand_points(n_rand_points: int, n_best_points: int, dataset: WeightedMolTreeFolder):
    chosen_point_set = set()

    if len(dataset.data) < n_best_points + n_rand_points:
        n_best_points, n_rand_points = int(n_best_points / (n_best_points + n_rand_points) * len(dataset.data)), int(
            n_rand_points / (n_best_points + n_rand_points) * len(dataset.data))
        n_rand_points += 1 if n_best_points + n_rand_points < len(dataset.data) else 0
    print(f"Take {n_best_points} best points and {n_rand_points} random points")

    # Best scores at start
    targets_argsort = np.argsort(-dataset.data_properties.flatten())
    for i in range(n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=n_rand_points + n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (n_rand_points + n_best_points):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (n_rand_points + n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_mol_trees(model, mol_trees):
    batch_size = 64
    mu_list = []
    with torch.no_grad():
        for i in trange(
                0, len(mol_trees), batch_size, desc="encoding GP points", leave=False
        ):
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    return mu


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu, store_best=False,
                  best_ckpt_path: Optional[str] = None):
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
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=True,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar, checkpointer],
        gradient_clip_val=20.0,  # Model is prone to large gradients
    )

    # Fit model
    trainer.fit(model, datamodule)

    if store_best:
        assert best_ckpt_path is not None
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
        shutil.copyfile(checkpointer.best_model_path, best_ckpt_path)


def get_root_path(lso_strategy: str, weight_type, k, r,
                  predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
                  beta_target_pred_loss: float,
                  metric_loss: str, metric_loss_kw: Dict[str, Any],
                  acq_func_id: str, acq_func_kwargs: Dict[str, Any],
                  input_wp: bool,
                  random_search_type: Optional[str],
                  use_pretrained: bool, pretrained_model_id: str, batch_size: int,
                  n_init_retrain_epochs: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int]
                  ):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO with semi-supervised training

    Returns:
        path to result dir
    """
    result_path = os.path.join(
        get_storage_root(),
        f"logs/opt/chem/{weight_type}/k_{k}/r_{r}")

    exp_spec = f"paper-mol"
    exp_spec += f'-z_dim_{latent_dim}'
    exp_spec += f"-init_{n_init_retrain_epochs:g}"
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
        exp_spec += f'-b_{float(beta_metric_loss):g}'
    exp_spec += f'-bkl_{beta_kl_final}'
    if semi_supervised:
        assert n_init_bo_points is not None, n_init_bo_points
        exp_spec += "-semi_supervised"
        exp_spec += f"-n-init-{n_init_bo_points}"
    if use_pretrained:
        exp_spec += f'_pretrain-{pretrained_model_id}'
    else:
        exp_spec += f'_scratch'
    if batch_size != 32:
        exp_spec += f'_bs-{batch_size}'

    if lso_strategy == 'opt':
        acq_func_spec = ''
        if acq_func_id != 'ExpectedImprovement':
            acq_func_spec += acq_func_id

        acq_func_spec += f"{'_inwp_' if input_wp else str(input_wp)}" \
            # if 'ErrorAware' in acq_func_id and cost_aware_gamma_sched is not None:
        #     acq_func_spec += f"_sch-{cost_aware_gamma_sched}"
        if len(acq_func_kwargs) > 0:
            acq_func_spec += f'_{str_dict(acq_func_kwargs)}'
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


def get_path(lso_strategy: str, weight_type, k, r,
             predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
             beta_target_pred_loss: float,
             metric_loss: str, metric_loss_kw: Dict[str, Any],
             acq_func_id: str, acq_func_kwargs: Dict[str, Any],
             input_wp: bool,
             random_search_type: Optional[str],
             use_pretrained: bool, pretrained_model_id: str, batch_size: int,
             n_init_retrain_epochs: int, seed: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int]):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        seed: for reproducibility
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO
    Returns:
        path to result dir
    """
    result_path = get_root_path(
        lso_strategy=lso_strategy,
        weight_type=weight_type,
        k=k,
        r=r,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        metric_loss=metric_loss,
        metric_loss_kw=metric_loss_kw,
        acq_func_id=acq_func_id,
        acq_func_kwargs=acq_func_kwargs,
        input_wp=input_wp,
        random_search_type=random_search_type,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_metric_loss=beta_metric_loss,
        beta_kl_final=beta_kl_final,
        use_pretrained=use_pretrained,
        n_init_retrain_epochs=n_init_retrain_epochs,
        batch_size=batch_size,
        semi_supervised=semi_supervised,
        n_init_bo_points=n_init_bo_points,
        pretrained_model_id=pretrained_model_id
    )
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.register('type', dict, parse_dict)

    parser = add_common_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    parser = wr_base.add_gp_args(parser)

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
        default=56,
        help="Hidden dimension the latent space",
    )
    parser.add_argument(
        "--use_pretrained",
        action='store_true',
        help="True if using pretrained VAE model",
    )
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default='vanilla',
        help="id of the pretrained VAE model used (should be aligned with the pretrained model file)",
    )

    vae_group = parser.add_argument_group("Metric learning")
    vae_group.add_argument(
        "--metric_loss",
        type=str,
        help="Metric loss to add to VAE loss during training of the generative model to get better "
             "structured latent space (see `METRIC_LOSSES`), one of ['contrastive', 'triplet', 'log_ratio', 'infob']",
    )
    vae_group.add_argument(
        "--metric_loss_kw",
        type=dict,
        default=None,
        help="Threshold parameter for contrastive loss, one of [{'threshold':.1}, {'threshold':.1,'margin':1}]",
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
        "--beta_start",
        type=float,
        default=None,
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
        help="Number of data points to use at the start of the BO if using semi-supervised training of VAE."
             "(We need at least SOME data to fit the GP(s) etc.)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0007,
        help="learning rate of the VAE training optimizer if needed (e.g. in case VAE from scratch)",
    )
    parser.add_argument(
        "--train-only",
        action='store_true',
        help="Train the JTVAE without running the BO",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="If `train-only`, save the trained model in save_model_path.",
    )
    args = parser.parse_args()

    args.train_path = os.path.join(ROOT_PROJECT, args.train_path)
    args.val_path = os.path.join(ROOT_PROJECT, args.val_path)
    args.vocab_file = os.path.join(ROOT_PROJECT, args.vocab_file)
    args.property_file = os.path.join(ROOT_PROJECT, args.property_file)

    if 'ErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
        assert 'eta' in args.acq_func_kwargs
        args.error_aware_acquisition = True
    else:
        args.error_aware_acquisition = False

    if args.pretrained_model_file is None:
        if args.use_pretrained:
            raise ValueError("You should specify the path to the pretrained model you want to use via "
                             "--pretrained_model_file argument")

    # Seeding
    pl.seed_everything(args.seed)

    # create result directory
    result_dir = get_path(
        lso_strategy=args.lso_strategy,
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        r=args.retraining_frequency,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        metric_loss=args.metric_loss,
        metric_loss_kw=args.metric_loss_kw,
        input_wp=args.input_wp,
        seed=args.seed,
        random_search_type=args.random_search_type,
        beta_metric_loss=args.beta_metric_loss,
        beta_target_pred_loss=args.beta_target_pred_loss,
        beta_kl_final=args.beta_final,
        use_pretrained=args.use_pretrained,
        n_init_retrain_epochs=args.n_init_retrain_epochs,
        semi_supervised=args.semi_supervised,
        n_init_bo_points=args.n_init_bo_points,
        pretrained_model_id=args.pretrained_model_id,
        batch_size=args.batch_size,
        acq_func_id=args.acq_func_id,
        acq_func_kwargs=args.acq_func_kwargs,
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
    f.write('\n' + '--------' * 10)
    f.write(logs)
    f.write('\n' + '--------' * 10)
    f.close()
    if exc is not None:
        raise exc


def main_aux(args, result_dir: str):
    """ main """

    # Seeding
    pl.seed_everything(args.seed)

    if args.train_only and os.path.exists(args.save_model_path) and not args.overwrite:
        print_flush(f'--- JTVAE already trained in {args.save_model_path} ---')
        return

    # Make results directory
    data_dir = os.path.join(result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    setup_logger(os.path.join(result_dir, "log.txt"))

    # Load data
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # Load model
    if args.use_pretrained:
        if args.predict_target:
            if 'pred_y' in args.pretrained_model_file:
                # fully supervised training from a model already trained with target prediction
                ckpt = torch.load(args.pretrained_model_file)
                ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
                ckpt['hyper_parameters']['hparams'].predict_target = True
                ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
                torch.save(ckpt, args.pretrained_model_file)
        print(os.path.abspath(args.pretrained_model_file))
        vae: JTVAE = JTVAE.load_from_checkpoint(args.pretrained_model_file, vocab=datamodule.vocab)
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
        vae.target_predictor_hdims = args.target_predictor_hdims
        vae.hparams.target_predictor_hdims = args.target_predictor_hdims
        if vae.predict_target and vae.target_predictor is None:
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
            vae.build_target_predictor()
    else:
        print("initialising VAE from scratch !")
        vae: JTVAE = JTVAE(hparams=args, vocab=datamodule.vocab)
    vae.eval()

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=float(datamodule.train_dataset.data_properties.max()),
        n_train=len(datamodule.train_dataset.data),
        save_path=result_dir
    )

    start_num_retrain = 0

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
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
        start_num_retrain = results['opt_model_version'][-1] + 1

        prev_retrain_model = args.retraining_frequency * (start_num_retrain - 1)
        num_sampled_points = len(results['opt_points'])
        if args.n_init_retrain_epochs == 0 and prev_retrain_model == 0:
            pretrained_model_path = args.pretrained_model_file
        else:
            pretrained_model_path = os.path.join(result_dir, 'retraining', f'retrain_{prev_retrain_model}',
                                                 'checkpoints',
                                                 'last.ckpt')
        print(f"Found checkpoint at {pretrained_model_path}")
        ckpt = torch.load(pretrained_model_path)
        ckpt['hyper_parameters']['hparams'].metric_loss = args.metric_loss
        ckpt['hyper_parameters']['hparams'].metric_loss_kw = args.metric_loss_kw
        ckpt['hyper_parameters']['hparams'].beta_metric_loss = args.beta_metric_loss
        ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
        if args.predict_target:
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
        torch.save(ckpt, pretrained_model_path)
        print(f"Loading model from {pretrained_model_path}")
        vae.load_from_checkpoint(pretrained_model_path, vocab=datamodule.vocab)
        if args.predict_target and not hasattr(vae.hparams, 'predict_target'):
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
        # vae.hparams.cuda = args.cuda
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.eval()

        # Set up some stuff for the progress bar
        num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency)) - start_num_retrain

        print(f"Append existing points and properties to datamodule...")
        datamodule.append_train_data(
            np.array(results['opt_points']),
            np.array(results['opt_point_properties'])
        )
        postfix = dict(
            retrain_left=num_retrain,
            best=float(datamodule.train_dataset.data_properties.max()),
            n_train=len(datamodule.train_dataset.data),
            initial=num_sampled_points,
            save_path=result_dir
        )
        print(f"Retrain from {result_dir} | Best: {max(results['opt_point_properties'])}")
    start_time = time.time()

    # Main loop
    with tqdm(
            total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(start_num_retrain, start_num_retrain + num_retrain):

            if vae.predict_target and vae.metric_loss is not None:
                vae.training_m = datamodule.training_m
                vae.training_M = datamodule.training_M
                vae.validation_m = datamodule.validation_m
                vae.validation_M = datamodule.validation_M

            torch.cuda.empty_cache()  # Free the memory up for tensorflow
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
                    version_str=version, num_epochs=num_epochs, gpu=args.gpu, store_best=args.train_only,
                    best_ckpt_path=args.save_model_path
                )
                vae.eval()
                if args.train_only:
                    return
            del num_epochs

            model = vae

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Draw samples for logs!
            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                with trange(
                        args.samples_per_model, desc="sampling", leave=False
                ) as sample_pbar:
                    sample_x, sample_y = latent_sampling(
                        args, model, datamodule, args.samples_per_model,
                        pbar=sample_pbar
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
            if args.lso_strategy == "opt":
                gp_dir = os.path.join(result_dir, "gp", f"iter{samples_so_far}")
                os.makedirs(gp_dir, exist_ok=True)
                gp_data_file = os.path.join(gp_dir, "data.npz")
                gp_err_data_file = os.path.join(gp_dir, "data_err.npz")
                x_new, y_new = latent_optimization(
                    model=model,
                    datamodule=datamodule,
                    n_inducing_points=args.n_inducing_points,
                    n_best_points=args.n_best_points,
                    n_rand_points=args.n_rand_points,
                    num_queries_to_do=num_queries_to_do,
                    gp_data_file=gp_data_file,
                    gp_err_data_file=gp_err_data_file,
                    gp_run_folder=gp_dir,
                    gpu=args.gpu,
                    invalid_score=args.invalid_score,
                    pbar=pbar,
                    postfix=postfix,
                    error_aware_acquisition=args.error_aware_acquisition,
                )
            elif args.lso_strategy == "sample":
                x_new, y_new = latent_sampling(
                    args, model, datamodule, num_queries_to_do, pbar=pbar,
                )
            else:
                raise NotImplementedError(args.lso_strategy)

            # Update dataset
            datamodule.append_train_data(x_new, y_new)

            # Add new results
            results["opt_points"] += list(x_new)
            results["opt_point_properties"] += list(y_new)
            results["opt_model_version"] += [ret_idx] * len(x_new)

            postfix["best"] = max(postfix["best"], float(max(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset.data)
            pbar.set_postfix(postfix)

            # Save results
            np.savez_compressed(os.path.join(result_dir, "results.npz"), **results)

            # Keep a record of the dataset here
            new_data_file = os.path.join(
                data_dir, f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
            )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.train_dataset.canonic_smiles))

    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


def latent_optimization(
        model: JTVAE,
        datamodule: WeightedJTNNDataset,
        n_inducing_points: int,
        n_best_points: int,
        n_rand_points: int,
        num_queries_to_do: int,
        invalid_score: float,
        gp_data_file: str,
        gp_run_folder: str,
        gpu: bool,
        error_aware_acquisition: bool,
        gp_err_data_file: Optional[str],
        pbar=None,
        postfix=None,
):
    ##################################################
    # Prepare GP
    ##################################################

    # First, choose GP points to train!
    dset = datamodule.train_dataset

    chosen_indices = _choose_best_rand_points(n_rand_points=n_rand_points, n_best_points=n_best_points, dataset=dset)
    mol_trees = [dset.data[i] for i in chosen_indices]
    targets = dset.data_properties[chosen_indices]
    chosen_smiles = [dset.canonic_smiles[i] for i in chosen_indices]

    # Next, encode these mol trees
    if gpu:
        model = model.cuda()
    latent_points = _encode_mol_trees(model, mol_trees)
    model = model.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()  # Free the memory up for tensorflow

    # Save points to file
    def _save_gp_data(x, y, s, file, flip_sign=True):

        # Prevent overfitting to bad points
        y = np.maximum(y, invalid_score)
        if flip_sign:
            y = -y.reshape(-1, 1)  # Since it is a maximization problem
        else:
            y = y.reshape(-1, 1)

        # Save the file
        np.savez_compressed(
            file,
            X_train=x.astype(np.float32),
            X_test=[],
            y_train=y.astype(np.float32),
            y_test=[],
            smiles=s,
        )

    # If using error-aware acquisition, compute reconstruction error of selected points
    if error_aware_acquisition:
        assert gp_err_data_file is not None, "Please provide a data file for the error GP"
        if gpu:
            model = model.cuda()
        error_train, safe_idx = get_rec_x_error(
            model,
            tkwargs={'dtype': torch.float},
            data=[datamodule.train_dataset.data[i] for i in chosen_indices],
        )
        # exclude points for which we could not compute the reconstruction error from the objective GP dataset
        if len(safe_idx) < latent_points.shape[0]:
            failed = [i for i in range(latent_points.shape[0]) if i not in safe_idx]
            print_flush(f"Could not compute the recon. err. of {len(failed)} points -> excluding them.")
            latent_points_err = latent_points[safe_idx]
            chosen_smiles_err = [chosen_smiles[i] for i in safe_idx]
        else:
            latent_points_err = latent_points
            chosen_smiles_err = chosen_smiles
        model = model.cpu()  # Make sure to free up GPU memory
        torch.cuda.empty_cache()  # Free the memory up for tensorflow
        _save_gp_data(latent_points, error_train.cpu().numpy(), chosen_smiles, gp_err_data_file)
    _save_gp_data(latent_points, targets, chosen_smiles, gp_data_file, flip_sign=False)

    ##################################################
    # Run iterative GP fitting/optimization
    ##################################################
    curr_gp_file = None
    curr_gp_err_file = None
    all_new_smiles = []
    all_new_props = []
    all_new_err = []

    for gp_iter in range(num_queries_to_do):
        gp_initial_train = gp_iter == 0
        current_n_inducing_points = min(latent_points.shape[0], n_inducing_points)
        if latent_points.shape[0] == n_inducing_points:
            gp_initial_train = True

        # Part 1: fit GP
        # ===============================
        new_gp_file = os.path.join(gp_run_folder, f"gp_train_res{gp_iter:04d}.npz")
        new_gp_err_file = os.path.join(gp_run_folder, f"gp_err_train_res0000.npz")  # no incremental fit of error-GP
        log_path = os.path.join(gp_run_folder, f"gp_train{gp_iter:04d}.log")
        err_log_path = os.path.join(gp_run_folder, f"gp_err_train0000.log")
        try:
            iter_seed = int(np.random.randint(10000))
            gp_train_command = [
                "python",
                GP_TRAIN_FILE,
                f"--nZ={current_n_inducing_points}",
                f"--seed={iter_seed}",
                f"--data_file={str(gp_data_file)}",
                f"--save_file={str(new_gp_file)}",
                f"--logfile={str(log_path)}",
                f"--normal_inputs",
                f"--standard_targets"
            ]
            gp_err_train_command = [
                "python",
                GP_TRAIN_FILE,
                f"--nZ={n_inducing_points}",
                f"--seed={iter_seed}",
                f"--data_file={str(gp_err_data_file)}",
                f"--save_file={str(new_gp_err_file)}",
                f"--logfile={str(err_log_path)}",
            ]
            if gp_initial_train:

                # Add commands for initial fitting
                gp_fit_desc = "GP initial fit"
                gp_train_command += [
                    "--init",
                    "--kmeans_init",
                ]
                gp_err_train_command += [
                    "--init",
                    "--kmeans_init",
                ]
            else:
                gp_fit_desc = "GP incremental fit"
                gp_train_command += [
                    f"--gp_file={str(curr_gp_file)}",
                    f"--n_perf_measure=1",  # specifically see how well it fits the last point!
                ]
                gp_err_train_command += [
                    f"--gp_file={str(curr_gp_err_file)}",
                    f"--n_perf_measure=1",  # specifically see how well it fits the last point!
                ]

            # Set pbar status for user
            if pbar is not None:
                old_desc = pbar.desc
                pbar.set_description(gp_fit_desc)

            # Run command
            print_flush("Training objective GP...")
            _run_command(gp_train_command, f"GP train {gp_iter}")
            curr_gp_file = new_gp_file
            if error_aware_acquisition:
                if gp_initial_train:  # currently we do not incrementally refit this GP as we do not estimate rec. err.
                    _run_command(gp_err_train_command, f"GP err train {gp_iter}")
                    curr_gp_err_file = new_gp_err_file
        except AssertionError as e:
            logs = traceback.format_exc()
            print(logs)
            print_flush(f'Got an error in GP training. Retrying with different seed or crash...')
            iter_seed = int(np.random.randint(10000))
            gp_train_command = [
                "python",
                GP_TRAIN_FILE,
                f"--nZ={current_n_inducing_points}",
                f"--seed={iter_seed}",
                f"--data_file={str(gp_data_file)}",
                f"--save_file={str(new_gp_file)}",
                f"--logfile={str(log_path)}",
            ]
            gp_err_train_command = [
                "python",
                GP_TRAIN_FILE,
                f"--nZ={n_inducing_points}",
                f"--seed={iter_seed}",
                f"--data_file={str(gp_err_data_file)}",
                f"--save_file={str(new_gp_err_file)}",
                f"--logfile={str(err_log_path)}",
                f"--normal_inputs",
                f"--standard_targets"
            ]
            if gp_initial_train:

                # Add commands for initial fitting
                gp_fit_desc = "GP initial fit"
                gp_train_command += [
                    "--init",
                    "--kmeans_init",
                ]
                gp_err_train_command += [
                    "--init",
                    "--kmeans_init",
                ]
            else:
                gp_fit_desc = "GP incremental fit"
                gp_train_command += [
                    f"--gp_file={str(curr_gp_file)}",
                    f"--n_perf_measure=1",  # specifically see how well it fits the last point!
                ]
                gp_err_train_command += [
                    f"--gp_file={str(curr_gp_err_file)}",
                    f"--n_perf_measure=1",  # specifically see how well it fits the last point!
                ]

            # Set pbar status for user
            if pbar is not None:
                old_desc = pbar.desc
                pbar.set_description(gp_fit_desc)

            # Run command
            _run_command(gp_train_command, f"GP train {gp_iter}")
            curr_gp_file = new_gp_file
            if error_aware_acquisition:
                if gp_initial_train:  # currently we do not incrementally refit this GP as we do not estimate rec. err.
                    _run_command(gp_err_train_command, f"GP err train {gp_iter}")
                    curr_gp_err_file = new_gp_err_file

        # Part 2: optimize GP acquisition func to query point
        # ===============================

        max_retry = 3
        n_retry = 0
        good = False
        while not good:
            try:
                # Run GP opt script
                opt_path = os.path.join(gp_run_folder, f"gp_opt_res{gp_iter:04d}.npy")
                log_path = os.path.join(gp_run_folder, f"gp_opt_{gp_iter:04d}.log")
                gp_opt_command = [
                    "python",
                    GP_OPT_FILE,
                    f"--seed={iter_seed}",
                    f"--gp_file={str(curr_gp_file)}",
                    f"--data_file={str(gp_data_file)}",
                    f"--save_file={str(opt_path)}",
                    f"--n_out={1}",  # hard coded
                    f"--logfile={str(log_path)}",
                ]
                if error_aware_acquisition:
                    gp_opt_command += [
                        f"--gp_err_file={str(curr_gp_err_file)}",
                        f"--data_err_file={str(gp_err_data_file)}",
                    ]

                if pbar is not None:
                    pbar.set_description("optimizing acq func")
                print_flush("Start running gp_opt_command")
                _run_command(gp_opt_command, f"GP opt {gp_iter}")

                # Load point
                z_opt = np.load(opt_path)

                # Decode point
                smiles_opt, prop_opt = _batch_decode_z_and_props(
                    model,
                    torch.as_tensor(z_opt, device=model.device),
                    datamodule,
                    invalid_score=invalid_score,
                    pbar=pbar,
                )
                good = True
            except AssertionError:
                iter_seed = int(np.random.randint(10000))
                n_retry += 1
                print_flush(f'Got an error in optimization......trial {n_retry} / {max_retry}')
                if n_retry >= max_retry:
                    raise

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = max(postfix["best"], float(max(prop_opt)))
                pbar.set_postfix(postfix)

        # Append to new GP data
        latent_points = np.concatenate([latent_points, z_opt], axis=0)
        targets = np.concatenate([targets, prop_opt], axis=0)
        chosen_smiles.append(smiles_opt)
        _save_gp_data(latent_points, targets, chosen_smiles, gp_data_file)

        # Append to overall list
        all_new_smiles += smiles_opt
        all_new_props += prop_opt

        if error_aware_acquisition:
            pass

    # Update datamodule with ALL data points
    return all_new_smiles, all_new_props


def latent_sampling(args, model, datamodule, num_queries_to_do, pbar=None):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )

    return z_decode, z_prop


if __name__ == "__main__":
    # Otherwise decoding fails completely
    rdkit_quiet()

    # Pytorch lightning raises some annoying unhelpful warnings
    # in this script (because it is set up weirdly)
    # therefore we suppress warnings
    # warnings.filterwarnings("ignore")

    main()
