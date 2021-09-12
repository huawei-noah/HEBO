import pytorch_lightning as pl
from torch.utils.data import TensorDataset, WeightedRandomSampler

NUM_WORKERS = 0

from torch.utils.data.dataloader import DataLoader, _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import _utils

from torchvision import transforms as transforms
import numpy as np
import numbers
from collections.abc import Sequence
from typing import List, Optional

import torch
from torch import Tensor


class DataLoaderTransformed(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None):
        super(DataLoaderTransformed, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                                    batch_sampler=batch_sampler, num_workers=num_workers,
                                                    collate_fn=collate_fn,
                                                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                                    worker_init_fn=worker_init_fn,
                                                    multiprocessing_context=multiprocessing_context,
                                                    generator=generator)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterModified(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


class _SingleProcessDataLoaderIterModified(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIterModified, self).__init__(loader)

        self.jitter_strength = 1.0

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            # transforms.RandomResizedCrop(size=self.input_height),
            # transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomGrayscale(p=0.2)
        ]

        kernel_size = int(0.1 * 64)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms.append(GaussianBlurAdaptive(kernel_size=kernel_size, p=0.5))
        # data_transforms.append(transforms.ToTensor())
        self.transform = transforms.Compose(data_transforms)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return self.transform(data[0]), self.transform(data[0])


class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the image by given kernel.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.
            In torchscript mode kernel_size as single int is not supported, use a sequence of length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None. In torchscript mode sigma as single float is
            not supported, use a sequence of length 1: ``[sigma, ]``.
    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))

    return img


class GaussianBlurAdaptive(object):
    # Implements Gaussian blur as described in the SimCLR paper with binary noise
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            blur = GaussianBlur(self.kernel_size, sigma)
            sample = blur(sample)

        return sample.clamp(min=0, max=1).round().float()


class WeightedNumpyDataset(pl.LightningDataModule):
    """ Implements a weighted numpy dataset (used for shapes task) """

    def __init__(self, hparams, data_weighter, add_channel: bool = True, clr=False):
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
        self.clr = clr
        self.data_weighter = data_weighter
        self.add_channel: bool = add_channel

        if not hasattr(hparams, 'predict_target'):
            hparams.predict_target = False
        self.predict_target: bool = hparams.predict_target
        self.maximize = True  # for the shapes task we want to minimize

        if not hasattr(hparams, 'metric_loss'):
            hparams.metric_loss = None
        self.metric_loss = hparams.metric_loss

        assert self.metric_loss is None or not self.predict_target, "Cannot handle both metric loss and target predictopm"

    def dataset_target_preprocess(self, targets: np.ndarray) -> Optional[np.ndarray]:
        """ Depending on the configuration, Dataloader should provide (normalized) targets """
        if self.predict_target:
            return targets / np.prod(self.data_train.shape[-2:])  # normalize the score by the image surface
        if self.metric_loss is not None:
            if self.metric_loss == 'contrastive':
                m, M = targets.min(), targets.max()
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

    def setup(self, stage=None):

        with np.load(self.dataset_path) as npz:
            all_data = npz["data"]
            all_properties = npz[self.property_key]
        assert all_properties.shape[0] == all_data.shape[0]

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
        self.train_dataset = self._get_tensor_dataset(self.data_train,
                                                      targets=self.dataset_target_preprocess(self.prop_train))
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
            self.train_data = np.concatenate(
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
        self.train_dataset = self._get_tensor_dataset(self.data_train,
                                                      targets=self.dataset_target_preprocess(self.prop_train))

    def train_dataloader(self):
        if self.clr:
            return DataLoaderTransformed(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKERS,
                sampler=self.train_sampler,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKERS,
                sampler=self.train_sampler,
                drop_last=True,
            )

    def val_dataloader(self):
        if self.clr:
            return DataLoaderTransformed(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKERS,
                sampler=self.val_sampler,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKERS,
                sampler=self.val_sampler,
                drop_last=True,
            )
