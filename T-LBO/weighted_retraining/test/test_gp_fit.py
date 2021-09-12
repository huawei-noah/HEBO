import argparse
import logging
import os
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

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
if os.path.join(ROOT_PROJECT, 'weighted_retraining') in sys.path:
    sys.path.remove(os.path.join(ROOT_PROJECT, 'weighted_retraining'))
sys.path[0] = ROOT_PROJECT

from weighted_retraining.weighted_retraining.utils import SubmissivePlProgressbar, DataWeighter, print_flush

from utils.utils_cmd import parse_list, parse_dict
from utils.utils_save import get_storage_root, save_w_pickle
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES

# My imports
from weighted_retraining.weighted_retraining import GP_TRAIN_FILE
from weighted_retraining.weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset,
    WeightedMolTreeFolder)
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet

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


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):
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


def get_root_path(weight_type, k,
                  predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
                  beta_target_pred_loss: float,
                  metric_loss: str, metric_loss_kw: Dict[str, Any],
                  input_wp: bool,
                  use_pretrained: bool, pretrained_model_id: str, batch_size: int,
                  n_init_retrain_epochs: float, n_test_points: int, use_decoded: float,
                  semi_supervised: Optional[bool], n_init_bo_points: Optional[int]
                  ):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        weight_type: type of weighting used for retraining
        k: weighting parameter
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO with semi-supervised setting
        n_test_points: number of test points on which gp fit will be evaluated
        use_decoded: whether to use f(x_test) or f(q(p(x_test))) as target for the gp

    Returns:
        path to result dir
    """
    result_path = os.path.join(
        get_storage_root(),
        f"logs/gp/chem/{weight_type}/k_{k}/")

    exp_spec = f"gp-fit"
    exp_spec += f'-z_dim_{latent_dim}'
    exp_spec += f"-init_{n_init_retrain_epochs:g}"
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
        exp_spec += f'-b_{float(beta_metric_loss):g}'
    if input_wp:
        exp_spec += f'-iw'
    exp_spec += f'-bkl_{beta_kl_final}'
    if semi_supervised:
        assert n_init_bo_points is not None, n_init_bo_points
        exp_spec += "-semi_supervised"
        exp_spec += f"-n-init-{n_init_bo_points}"
    if use_pretrained:
        if pretrained_model_id != 'vanilla':
            exp_spec += f'_pretrain-{pretrained_model_id}'
    exp_spec += f'_bs-{batch_size}'
    result_path = os.path.join(result_path, exp_spec, f"{n_test_points}" + ("-dec" if use_decoded else ""))

    return result_path


def get_path(weight_type, k,
             predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
             beta_target_pred_loss: float,
             metric_loss: str, metric_loss_kw: Dict[str, Any],
             input_wp: bool, use_pretrained: bool, pretrained_model_id: str, batch_size: int,
             n_init_retrain_epochs: int, n_test_points: int, use_decoded: bool,
             seed: float, semi_supervised: Optional[bool],
             n_init_bo_points: Optional[int]):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        seed: for reproducibility
        weight_type: type of weighting used for retraining
        k: weighting parameter
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO
        n_test_points: number of test points on which gp fit will be evaluated
        use_decoded: whether to use f(x_test) or f(q(p(x_test))) as target for the gp
    Returns:
        path to result dir
    """
    result_path = get_root_path(
        weight_type=weight_type,
        k=k,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        metric_loss=metric_loss,
        metric_loss_kw=metric_loss_kw,
        input_wp=input_wp,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_metric_loss=beta_metric_loss,
        beta_kl_final=beta_kl_final,
        use_pretrained=use_pretrained,
        n_init_retrain_epochs=n_init_retrain_epochs,
        batch_size=batch_size,
        semi_supervised=semi_supervised,
        n_init_bo_points=n_init_bo_points,
        n_test_points=n_test_points,
        use_decoded=use_decoded,
        pretrained_model_id=pretrained_model_id
    )
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


def add_gp_fit_args(parser: argparse.ArgumentParser):
    gp_group = parser.add_argument_group("Sparse GP")
    gp_group.add_argument("--n_inducing_points", type=int, default=500)
    gp_group.add_argument("--n_rand_points", type=int, default=8000)
    gp_group.add_argument("--n_best_points", type=int, default=2000)
    gp_group.add_argument("--invalid_score", type=float, default=-4.0)
    return parser


def add_main_args(parser: argparse.ArgumentParser):
    opt_group = parser.add_argument_group("weighted retraining")
    opt_group.add_argument("--seed", type=int, required=True)
    opt_group.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    opt_group.add_argument("--result_root", type=str, help="root directory to store results in")
    opt_group.add_argument("--pretrained_model_file", type=str, default=None,
                           help="path to pretrained model to use")
    opt_group.add_argument("--version", type=int, default=None,
                           help="Version of the model (not required if `pretrained_model_file` is specified)")
    opt_group.add_argument("--n_init_retrain_epochs", type=float, default=None,
                           help="None to use n_retrain_epochs, 0.0 to skip init retrain")

    opt_group.add_argument("--overwrite", action='store_true',
                           help="Whether to overwrite existing results (that will be found in result dir) -"
                                " Default: False")
    return parser


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.register('type', dict, parse_dict)

    parser = add_main_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    parser = add_gp_fit_args(parser)

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
        "--n_test_points",
        type=int,
        default=2500,
        help="Number of held-out data points to use for gp fit assessment"
    )

    vae_group.add_argument(
        "--use_decoded",
        action='store_true',
        help="whether to use f(x_test) or f(q(p(x_test))) as test target for the gp"
    )
    args = parser.parse_args()

    args.train_path = os.path.join(ROOT_PROJECT, args.train_path)
    args.val_path = os.path.join(ROOT_PROJECT, args.val_path)
    args.vocab_file = os.path.join(ROOT_PROJECT, args.vocab_file)
    args.property_file = os.path.join(ROOT_PROJECT, args.property_file)

    if args.pretrained_model_file is not None:
        args.pretrained_model_file = os.path.join(get_storage_root(), args.pretrained_model_file)
    else:
        raise ValueError("does not support this yet, use pretrained model, please.")

    # create result directory
    result_dir = get_path(
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        metric_loss=args.metric_loss,
        metric_loss_kw=args.metric_loss_kw,
        input_wp=args.input_wp,
        seed=args.seed,
        beta_metric_loss=args.beta_metric_loss,
        beta_target_pred_loss=args.beta_target_pred_loss,
        beta_kl_final=args.beta_final,
        use_pretrained=args.use_pretrained,
        n_init_retrain_epochs=args.n_init_retrain_epochs,
        semi_supervised=args.semi_supervised,
        n_init_bo_points=args.n_init_bo_points,
        pretrained_model_id=args.pretrained_model_id,
        batch_size=args.batch_size,
        use_decoded=args.use_decoded,
        n_test_points=args.n_test_points,
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

    # Make results directory
    data_dir = os.path.join(result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    setup_logger(os.path.join(result_dir, "log.txt"))

    result_filepath = os.path.join(result_dir, 'results.pkl')
    if not args.overwrite and os.path.exists(result_filepath):
        print(f"Already exists: {result_dir}")
        return

    # Load data
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # Load model
    assert args.use_pretrained

    if args.predict_target:
        if 'pred_y' in args.pretrained_model_file:
            # fully supervised setup from a model trained with target prediction
            ckpt = torch.load(args.pretrained_model_file)
            ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
            torch.save(ckpt, args.pretrained_model_file)
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
    vae.eval()

    # Set up some stuff for the progress bar
    postfix = dict(
        n_train=len(datamodule.train_dataset.data),
        save_path=result_dir
    )

    # Set up results tracking
    start_time = time.time()

    train_chosen_indices = _choose_best_rand_points(n_rand_points=args.n_rand_points, n_best_points=args.n_best_points,
                                                    dataset=datamodule.train_dataset)
    train_mol_trees = [datamodule.train_dataset.data[i] for i in train_chosen_indices]
    train_targets = datamodule.train_dataset.data_properties[train_chosen_indices]
    train_chosen_smiles = [datamodule.train_dataset.canonic_smiles[i] for i in train_chosen_indices]

    test_chosen_indices = _choose_best_rand_points(n_rand_points=args.n_test_points, n_best_points=0,
                                                   dataset=datamodule.val_dataset)
    test_mol_trees = [datamodule.val_dataset.data[i] for i in test_chosen_indices]

    # Main loop
    with tqdm(
            total=1, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        if vae.predict_target and vae.metric_loss is not None:
            vae.training_m = datamodule.training_m
            vae.training_M = datamodule.training_M
            vae.validation_m = datamodule.validation_m
            vae.validation_M = datamodule.validation_M

        torch.cuda.empty_cache()  # Free the memory up for tensorflow
        pbar.set_postfix(postfix)
        pbar.set_description("retraining")
        print(result_dir)

        # Optionally do retraining
        num_epochs =  args.n_init_retrain_epochs
        if num_epochs > 0:
            retrain_dir = os.path.join(result_dir, "retraining")
            version = f"retrain_0"
            retrain_model(
                model=vae, datamodule=datamodule, save_dir=retrain_dir,
                version_str=version, num_epochs=num_epochs, gpu=args.gpu
            )
            vae.eval()
        del num_epochs

        model = vae

        # Update progress bar
        pbar.set_postfix(postfix)

        # Do querying!
        gp_dir = os.path.join(result_dir, "gp")
        os.makedirs(gp_dir, exist_ok=True)
        gp_data_file = os.path.join(gp_dir, "data.npz")

        # Next, encode these mol trees
        if args.gpu:
            model = model.cuda()
        train_latent_points = _encode_mol_trees(model, train_mol_trees)
        test_latent_points = _encode_mol_trees(model, test_mol_trees)
        if args.use_decoded:
            print("Use targets from decoded latent test points")
            _, test_targets = _batch_decode_z_and_props(
                model,
                torch.as_tensor(test_latent_points, device=model.device),
                datamodule,
                invalid_score=args.invalid_score,
                pbar=pbar,
            )
            test_targets = np.array(test_targets)
        else:
            test_targets = datamodule.val_dataset.data_properties[test_chosen_indices]

        model = model.cpu()  # Make sure to free up GPU memory
        torch.cuda.empty_cache()  # Free the memory up for tensorflow

        # Save points to file
        def _save_gp_data(x, y, test_x, y_test, s, file, flip_sign=True):

            # Prevent overfitting to bad points
            y = np.maximum(y, args.invalid_score)
            if flip_sign:
                y = -y.reshape(-1, 1)  # Since it is a maximization problem
                y_test = -y_test.reshape(-1, 1)
            else:
                y = y.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

            # Save the file
            np.savez_compressed(
                file,
                X_train=x.astype(np.float32),
                X_test=test_x.astype(np.float32),
                y_train=y.astype(np.float32),
                y_test=y_test.astype(np.float32),
                smiles=s,
            )

        _save_gp_data(train_latent_points, train_targets, test_latent_points, test_targets, train_chosen_smiles,
                      gp_data_file)
        current_n_inducing_points = min(train_latent_points.shape[0], args.n_inducing_points)

        new_gp_file = os.path.join(gp_dir, f"new.npz")
        log_path = os.path.join(gp_dir, f"gp_fit.log")

        iter_seed = int(np.random.randint(10000))
        gp_train_command = [
            "python",
            GP_TRAIN_FILE,
            f"--nZ={current_n_inducing_points}",
            f"--seed={iter_seed}",
            f"--data_file={str(gp_data_file)}",
            f"--save_file={str(new_gp_file)}",
            f"--logfile={str(log_path)}",
            "--use_test_set"
        ]
        gp_fit_desc = "GP initial fit"
        gp_train_command += [
            "--init",
            "--kmeans_init",
            f"--save_metrics_file={str(result_filepath)}"
        ]
        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        _run_command(gp_train_command, f"GP train {0}")
        curr_gp_file = new_gp_file

    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


if __name__ == "__main__":
    # Otherwise decoding fails completely
    rdkit_quiet()

    # Pytorch lightning raises some annoying unhelpful warnings
    # in this script (because it is set up weirdly)
    # therefore we suppress warnings
    # warnings.filterwarnings("ignore")

    main()
