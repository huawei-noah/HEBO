""" Trains a convnet for the Topology task """
import argparse
import os
import shutil
import sys
from typing import Any, Optional, Dict, List
from pathlib import Path

from torchvision.transforms import transforms

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from weighted_retraining.weighted_retraining.topology.topology_dataset import get_topology_dataset_path, \
    gen_dataset_from_all_files, gen_binary_dataset_from_all_files, get_topology_binary_dataset_path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from utils.utils_save import get_storage_root, get_data_root
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from weighted_retraining.weighted_retraining.topology.topology_data import WeightedNumpyDataset
from weighted_retraining.weighted_retraining.topology.topology_model import TopologyVAE
from weighted_retraining.weighted_retraining.utils import print_flush


def topology_get_path(k, predict_target: bool, n_max_epochs: int,
                      beta_final: float, beta_target_pred_loss: float, beta_metric_loss: float,
                      latent_dim: int = 25, hdims: List[int] = None,
                      metric_loss: Optional[str] = None, metric_loss_kw: Dict[str, Any] = None,
                      use_binary_data: bool = False
                      ):
    """ Get path of directory where models will be stored

    Args:
        k: weight parameter
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure the embedding
        metric_loss_kw: kwargs for `metric_loss` (see `METRIC_LOSSES`)
use_binary_data: use binarized data

    Returns:
        Path to result dir
    """
    res_path = os.path.join(get_storage_root(), f'logs/train/topology/k-{k}')
    exp_spec = f'id'
    if latent_dim != 2:
        exp_spec += f'-z_dim_{latent_dim}'
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        if beta_target_pred_loss != 1:
            exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
        if beta_metric_loss != 1:
            exp_spec += f'-b_{float(beta_metric_loss):g}'
    exp_spec += f"-bkl_{beta_final}"
    if use_binary_data:
        exp_spec += '-binary_data'
    res_path = os.path.join(res_path, exp_spec)
    print('res_path', res_path)
    return res_path


def main():
    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = TopologyVAE.add_model_specific_args(parser)
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root="")

    parser.add_argument(
        "--augment_dataset",
        action='store_true',
        help="Use data augmentation or not",
    )
    parser.add_argument(
        "--use_binary_data",
        action='store_true',
        help="Binarize images in the dataset",
    )

    # Parse arguments
    hparams = parser.parse_args()
    hparams.root_dir = topology_get_path(k=hparams.rank_weight_k, n_max_epochs=hparams.max_epochs,
                                         predict_target=hparams.predict_target,
                                         hdims=hparams.target_predictor_hdims, metric_loss=hparams.metric_loss,
                                         metric_loss_kw=hparams.metric_loss_kw,
                                         beta_target_pred_loss=hparams.beta_target_pred_loss,
                                         beta_metric_loss=hparams.beta_metric_loss,
                                         latent_dim=hparams.latent_dim,
                                         beta_final=hparams.beta_final, use_binary_data=hparams.use_binary_data)
    print_flush(' '.join(sys.argv[1:]))
    print_flush(hparams.root_dir)
    pl.seed_everything(hparams.seed)

    # Create data
    if hparams.use_binary_data:
        if not os.path.exists(os.path.join(get_data_root(), 'topology_data/target_bin.npy')):
            gen_binary_dataset_from_all_files(get_data_root())
        hparams.dataset_path = os.path.join(ROOT_PROJECT, get_topology_binary_dataset_path())
    else:
        if not os.path.exists(os.path.join(get_data_root(), 'topology_data/target.npy')):
            gen_dataset_from_all_files(get_data_root())
        hparams.dataset_path = os.path.join(ROOT_PROJECT, get_topology_dataset_path())
    if hparams.augment_dataset:
        aug = transforms.Compose([
            # transforms.Normalize(mean=, std=),
            # transforms.RandomCrop(30, padding=10),
            transforms.RandomRotation(45),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(0.5)
        ])
    else:
        aug = None
    datamodule = WeightedNumpyDataset(hparams, utils.DataWeighter(hparams), transform=aug)

    # Load model
    model = TopologyVAE(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        period=max(1, hparams.max_epochs // 10),
        monitor="loss/val", save_top_k=-1,
        save_last=True, mode='min'
    )

    if hparams.load_from_checkpoint is not None:
        model = TopologyVAE.load_from_checkpoint(hparams.load_from_checkpoint)
        utils.update_hparams(hparams, model)
        trainer = pl.Trainer(gpus=[hparams.cuda] if hparams.cuda else 0,
                             default_root_dir=hparams.root_dir,
                             max_epochs=hparams.max_epochs,
                             callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
                             resume_from_checkpoint=hparams.load_from_checkpoint)

        print(f'Load from checkpoint')
    else:
        # Main trainer
        trainer = pl.Trainer(
            gpus=[hparams.cuda] if hparams.cuda is not None else 0,
            default_root_dir=hparams.root_dir,
            max_epochs=hparams.max_epochs,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
            terminate_on_nan=True,
            progress_bar_refresh_rate=5,
            # gradient_clip_val=20.0,
        )

    # Fit
    trainer.fit(model, datamodule=datamodule)

    print(f"Training finished; end of script: rename {checkpoint_callback.best_model_path}")

    shutil.copyfile(checkpoint_callback.best_model_path, os.path.join(
        os.path.dirname(checkpoint_callback.best_model_path), 'best.ckpt'
    ))


if __name__ == "__main__":
    main()
