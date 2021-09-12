""" Trains a VAE for the equation task """
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from utils.utils_save import get_storage_root
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.expr.expr_data_pt import WeightedExprDataset
from weighted_retraining.weighted_retraining.expr.expr_dataset import get_filepath
from weighted_retraining.weighted_retraining.expr.expr_model_pt import EquationVaeTorch
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from weighted_retraining.weighted_retraining.utils import print_flush
from typing import Any, Optional, Dict, List

MAX_LEN = 15
import weighted_retraining.weighted_retraining.expr.eq_grammar as G


def get_path(k, ignore_percentile, good_percentile, predict_target: bool, hdims: List[int] = None,
             metric_loss: Optional[str] = None, metric_loss_kw: Dict[str, Any] = None):
    """ Get path of directory where models will be stored

    Args:
        k: weight parameter
        ignore_percentile: portion of original equation dataset ignored
        good_percentile: portion of good original equation dataset included
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure the embedding
        metric_loss_kw: kwargs for `metric_loss` (see `METRIC_LOSSES`)

    Returns:
        Path to result dir
    """
    res_path = os.path.join(get_storage_root(), f'logs/train/expr/torch/expr-k-{k}')
    exp_spec = f'ignore_perc-{ignore_percentile}_good_perc-{good_percentile}'
    if predict_target:
        assert hdims is not None
        exp_spec += '_predy-' + '-'.join(map(str, hdims))
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
    res_path = os.path.join(res_path, exp_spec)
    print('res_path', res_path)
    return res_path


def main():
    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = EquationVaeTorch.add_model_specific_args(parser)
    parser = WeightedExprDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root='')

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
        "--data_seed",
        type=int,
        required=True,
        help="Seed that has been used to generate the dataset"
    )

    # Parse arguments
    hparams = parser.parse_args()
    hparams.dataset_path = get_filepath(hparams.ignore_percentile, hparams.dataset_path, hparams.data_seed,
                                        good_percentile=hparams.good_percentile)
    hparams.root_dir = get_path(k=hparams.rank_weight_k, ignore_percentile=hparams.ignore_percentile,
                                good_percentile=hparams.good_percentile, predict_target=hparams.predict_target,
                                hdims=hparams.target_predictor_hdims, metric_loss=hparams.metric_loss,
                                metric_loss_kw=hparams.metric_loss_kw)
    print_flush(' '.join(sys.argv[1:]))
    print_flush(hparams.root_dir)

    pl.seed_everything(hparams.seed)

    # Create data
    datamodule = WeightedExprDataset(hparams, utils.DataWeighter(hparams), add_channel=False)

    device = hparams.cuda
    if device is not None:
        torch.cuda.set_device(device)

    data_info = G.gram.split('\n')

    # Load model
    model = EquationVaeTorch(hparams, len(data_info), MAX_LEN)
    # model.decoder.apply(torch_weight_init)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        period=1, monitor="loss/val", save_top_k=1,
        save_last=True, mode='min'
    )

    if hparams.load_from_checkpoint is not None:
        # .load_from_checkpoint(hparams.load_from_checkpoint)
        model = EquationVaeTorch.load_from_checkpoint(hparams.load_from_checkpoint, len(data_info), MAX_LEN)
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
            progress_bar_refresh_rate=100
        )

    # Fit
    trainer.fit(model, datamodule=datamodule)

    print(f"Training finished; end of script: rename {checkpoint_callback.best_model_path}")

    shutil.copyfile(checkpoint_callback.best_model_path, os.path.join(
        os.path.dirname(checkpoint_callback.best_model_path), 'best.ckpt'
    ))


if __name__ == "__main__":
    main()
