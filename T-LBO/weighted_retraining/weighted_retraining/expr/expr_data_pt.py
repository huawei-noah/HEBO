from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from weighted_retraining.weighted_retraining.utils import print_flush

NUM_WORKERS = 3


class WeightedExprDataset(pl.LightningDataModule):
    """ Implements a weighted numpy dataset (used for expression task) """

    def __init__(self, hparams, data_weighter, add_channel: bool = False):
        """
        Args:
            hparams:
            data_weighter: what kind of data weighter to use ('uniform', 'rank'...)
            add_channel: whether to unsqueeze first dim when converting to tensor
                         (adding channel dimension for image dataset)
        """
        super().__init__()
        self.dataset_path: str = hparams.dataset_path
        self.val_frac: float = hparams.val_frac
        self.property_key: str = hparams.property_key
        self.expr_key: str = hparams.second_key
        self.batch_size: int = hparams.batch_size

        self.data_weighter = data_weighter
        self.add_channel: bool = add_channel

        self.predict_target: bool = hparams.predict_target
        self.maximize: bool = False  # for the expression task we want to minimize

        self.metric_loss = hparams.metric_loss

        assert self.metric_loss is None or not self.predict_target, "Cannot handle both metric loss and target prediction"

    def dataset_target_preprocess(self, targets: np.ndarray) -> Optional[np.ndarray]:
        """ Depending on the configuration, Dataloader should provide (normalized) targets """
        if self.predict_target:
            return targets
        if self.metric_loss is not None:
            if self.metric_loss in ('contrastive', 'triplet', 'log_ratio'):
                m, M = targets.min(), targets.max()
                if m == M:
                    return np.ones_like(targets)
                return (targets - m) / (M - m)
            else:
                raise ValueError(f'{self.metric_loss} not supported')
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument(
            "--dataset_path", type=str, required=True, help="path to npz file"
        )

        data_group.add_argument("--batch_size", type=int, default=512)
        data_group.add_argument(
            "--val_frac",
            type=float,
            default=0.05,
            help="Fraction of val data. Note that data is NOT shuffled!!!",
        )
        data_group.add_argument(
            "--property_key",
            type=str,
            default='scores',
            help="Key in npz file to the object properties",
        )
        data_group.add_argument(
            "--second_key",
            type=str,
            default='expr',
            help="Key in npz file to the object properties",
        )
        return parent_parser

    def prepare_data(self):
        pass

    def _get_tensor_dataset(self, data, targets=None) -> TensorDataset:
        data = torch.as_tensor(data, dtype=torch.float)
        if self.add_channel:
            data = torch.unsqueeze(data, 1)
        datas = [data]
        if targets is not None:
            targets = torch.as_tensor(targets, dtype=torch.float).unsqueeze(1)
            assert targets.ndim == 2, targets.shape
            datas.append(targets)
        return TensorDataset(*datas)

    def setup(self, stage=None, n_init_points: Optional[bool] = None):

        with np.load(self.dataset_path) as npz:
            all_data = npz["data"]
            all_properties = npz[self.property_key]
            all_exprs = npz[self.expr_key]

        self.expr_set = set(all_exprs)

        assert all_properties.shape[0] == all_data.shape[0] == all_exprs.shape[0]

        if n_init_points is not None:
            indices = np.random.randint(0, all_data.shape[0], n_init_points)
            all_data = all_data[indices]
            all_exprs = all_exprs[indices]
            all_properties = all_properties[indices]

        N_val = int(all_data.shape[0] * self.val_frac)
        self.data_val = all_data[:N_val]
        self.prop_val = all_properties[:N_val]
        self.expr_val = all_exprs[:N_val]

        self.data_train = all_data[N_val:]
        self.prop_train = all_properties[N_val:]
        self.expr_train = all_exprs[N_val:]
        self.set_weights()

        self.specific_setup()

    def specific_setup(self):
        # Make into tensor datasets
        self.train_dataset = self._get_tensor_dataset(self.data_train,
                                                      targets=self.dataset_target_preprocess(self.prop_train))
        self.val_dataset = self._get_tensor_dataset(self.data_val,
                                                    targets=self.dataset_target_preprocess(self.prop_val))

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(-self.prop_train)  # the lower the better
        self.val_weights = self.data_weighter.weighting_function(-self.prop_val)

        # Create samplers
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new, expr_new):

        # discard invalid (None) inputs and their corresponding scores
        print(f"Appending new expression to datamodule.", expr_new)
        valid_idx = ~(expr_new == None)
        valid_idx = np.array(valid_idx)
        print_flush(
            "Discarding {}/{} new inputs that are invalid!".format(len(valid_idx) - valid_idx.sum(), len(valid_idx))
        )
        expr_new = list(expr_new[valid_idx])
        prop_new = prop_new[valid_idx]
        x_new = x_new[valid_idx]


        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(-prop_new, self.data_weighter.weight_quantile)
            indices_to_add = (-prop_new >= cutoff)  # minimize prop

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            self.train_data = np.concatenate(
                [self.data_train[len(x_new):], x_new], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new):], prop_new], axis=0
            )
            self.expr_train = np.concatenate(
                [self.expr_train[len(x_new):], expr_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = np.concatenate([self.data_train, x_new], axis=0)
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
            self.expr_train = np.concatenate([self.expr_train, expr_new], axis=0)
        self.set_weights()
        self.append_train_data_specific()

    def append_train_data_specific(self):
        self.train_dataset = self._get_tensor_dataset(self.data_train,
                                                      targets=self.dataset_target_preprocess(self.prop_train))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=min(len(self.train_dataset), self.batch_size),
            num_workers=NUM_WORKERS,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(len(self.val_dataset), self.batch_size),
            num_workers=NUM_WORKERS,
            sampler=self.val_sampler,
            drop_last=True,
        )
