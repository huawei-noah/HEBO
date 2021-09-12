""" Script to train chem model """

import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[-1] = ROOT_PROJECT

import pytorch_lightning as pl

# My imports
from pytorch_lightning.callbacks import LearningRateMonitor

from utils.utils_save import get_storage_root
from weighted_retraining.weighted_retraining.utils import print_flush

from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.chem_data import WeightedJTNNDataset
from weighted_retraining.weighted_retraining import utils

if __name__ == "__main__":

    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = JTVAE.add_model_specific_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root=None)

    # Parse arguments
    hparams = parser.parse_args()

    hparams.root_dir = os.path.join(get_storage_root(), hparams.root_dir)

    pl.seed_everything(hparams.seed)
    print_flush(' '.join(sys.argv[1:]))

    # Create data
    datamodule = WeightedJTNNDataset(hparams, utils.DataWeighter(hparams))
    datamodule.setup("fit")

    # Load model
    model = JTVAE(hparams, datamodule.vocab)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        period=1, monitor="loss/val", save_top_k=1,
        save_last=True, mode='min'
    )

    if hparams.load_from_checkpoint is not None:
            # .load_from_checkpoint(hparams.load_from_checkpoint)
        # utils.update_hparams(hparams, model)
        trainer = pl.Trainer(gpus=[hparams.cuda] if hparams.cuda else 0,
            default_root_dir=hparams.root_dir,
            max_epochs=hparams.max_epochs,
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
            terminate_on_nan=True,
            gradient_clip_val=20.0, resume_from_checkpoint=hparams.load_from_checkpoint)

        print(f'Load from checkpoint')

    else:
        # Main trainer
        trainer = pl.Trainer(
            gpus=[hparams.cuda] if hparams.cuda else 0,
            default_root_dir=hparams.root_dir,
            max_epochs=hparams.max_epochs,
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
            terminate_on_nan=True,
            gradient_clip_val=20.0  # Model is prone to large gradients
        )

    # Fit
    trainer.fit(model, datamodule=datamodule)
    print("Training finished; end of script")
    os.rename(checkpoint_callback.best_model_path, os.path.join(
        os.path.dirname(checkpoint_callback.best_model_path), 'best.ckpt'
    ))
