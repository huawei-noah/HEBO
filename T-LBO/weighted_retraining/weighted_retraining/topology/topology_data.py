import pytorch_lightning as pl
from torch.utils.data import TensorDataset, WeightedRandomSampler, Dataset

NUM_WORKERS = 0

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import _utils

from torchvision import transforms as transforms
import numpy as np
import numbers
from collections.abc import Sequence
from typing import List, Optional, Tuple

import torch
from torch import Tensor


class AugmentedTensorDataset(TensorDataset):

    def __init__(self, *tensors, transform=None):
        super(AugmentedTensorDataset, self).__init__(*tensors)
        self.transform = transform

    def __getitem__(self, idx):
        selected = tuple(tensor[idx] for tensor in self.tensors)
        x = selected[0]
        if self.transform is not None:
            x = self.transform(x)
        if len(selected) == 1:
            return (x,)
        elif len(selected) == 2:
            y = selected[1]
            return x, y
        else:
            raise ValueError(f"Can only have two tensor groups in dataset, x and y but len(selected)={len(selected)}")

class WeightedNumpyDataset(pl.LightningDataModule):
    """ Implements a weighted numpy dataset (used for shapes task) """

    def __init__(self, hparams, data_weighter, add_channel: bool = True, transform=None):
        """

        Args:
            hparams:
            data_weighter: what kind of data weighter to use ('uniform', 'rank'...)
            add_channel: whether to unsqueeze first dim when converting to tensor
                         (adding channel dimension for image dataset)
        """
        super().__init__()
        self.dataset_path = hparams.dataset_path
        self.val_frac = hparams.val_frac
        self.property_key = hparams.property_key
        self.batch_size = hparams.batch_size
        self.data_weighter = data_weighter
        self.add_channel: bool = add_channel

        if not hasattr(hparams, 'predict_target'):
            hparams.predict_target = False
        self.predict_target: bool = hparams.predict_target
        self.maximize = True  # for the shapes task we want to minimize

        if not hasattr(hparams, 'metric_loss'):
            hparams.metric_loss = None
        self.metric_loss = hparams.metric_loss

        self.transform = transform

        assert self.metric_loss is None or not self.predict_target, "Cannot handle both metric loss and target predictopm"

    def dataset_target_preprocess(self, targets: np.ndarray) -> Optional[np.ndarray]:
        """ Depending on the configuration, Dataloader should provide targets """
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
        # data_group.add_argument(
        #     "--dataset_path", type=str, required=True, help="path to npz file"
        # )

        data_group.add_argument("--batch_size", type=int, default=64)
        data_group.add_argument(
            "--val_frac",
            type=float,
            default=0.05,
            help="Fraction of val data. Note that data is NOT shuffled!!!",
        )
        data_group.add_argument(
            "--property_key",
            type=str,
            required=True,
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

    def _get_augmented_tensor_dataset(self, data, targets=None, transform=None) -> AugmentedTensorDataset:
        data = torch.as_tensor(data, dtype=torch.float)
        if self.add_channel:
            data = torch.unsqueeze(data, 1)
        datas = [data]
        if targets is not None:
            targets = torch.as_tensor(targets, dtype=torch.float).unsqueeze(1)
            assert targets.ndim == 2, targets.shape
            datas.append(targets)
        return AugmentedTensorDataset(*datas, transform=transform)

    def setup(self, stage=None, n_init_points: Optional[bool] = None):

        with np.load(self.dataset_path) as npz:
            all_data = npz["data"]
            all_properties = npz[self.property_key]
        assert all_properties.shape[0] == all_data.shape[0]

        if n_init_points is not None:
            indices = np.random.randint(0, all_data.shape[0], n_init_points)
            all_data = all_data[indices]
            all_properties = all_properties[indices]

        N_val = int(all_data.shape[0] * self.val_frac)
        self.data_val = all_data[:N_val]
        self.prop_val = all_properties[:N_val]
        self.data_train = all_data[N_val:]
        self.prop_train = all_properties[N_val:]

        # Make into tensor datasets
        self.set_weights()
        self.specific_setup()

    def specific_setup(self):
        # Make into tensor datasets
        self.train_dataset = self._get_augmented_tensor_dataset(self.data_train,
                                                                targets=self.dataset_target_preprocess(self.prop_train),
                                                                transform=self.transform)
        self.val_dataset = self._get_tensor_dataset(self.data_val,
                                                    targets=self.dataset_target_preprocess(self.prop_val))

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(self.prop_train)
        self.val_weights = self.data_weighter.weighting_function(self.prop_val)

        # Create samplers
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new):

        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(prop_new, self.data_weighter.weight_quantile)
            indices_to_add = prop_new >= cutoff

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            self.data_train = np.concatenate(
                [self.data_train[len(x_new):], x_new], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new):], prop_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = np.concatenate([self.data_train, x_new], axis=0)
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
        self.set_weights()
        self.append_train_data_specific()

    def append_train_data_specific(self):
        self.train_dataset = self._get_augmented_tensor_dataset(self.data_train,
                                                                targets=self.dataset_target_preprocess(self.prop_train),
                                                                transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, len(self.train_dataset)),
            num_workers=NUM_WORKERS,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=NUM_WORKERS,
            sampler=self.val_sampler,
            drop_last=True,
        )
