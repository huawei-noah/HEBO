import numpy as np
import torch
import botorch.models.transforms as trans
from torch.nn import ModuleDict
from collections import OrderedDict
from typing import Tuple
from torch import Tensor
from botorch.exceptions.errors import BotorchTensorDimensionError
from typing import Any, Callable, Dict, List, Optional, Union


def additional_std(X, std):
    ret = torch.ones_like(X) * torch.tensor(std, dtype=X.dtype, device=X.device) \
        if isinstance(X, torch.Tensor) else np.ones_like(X) * std
    return ret


def additional_xc_samples(X, n_sample, n_var, sampling_func, sampling_cfg={}, **kwargs):
    # more samples around the raw
    batch_shape = X.shape[:-1]
    noises = sampling_func(
        **{**sampling_cfg, 'x': X}, size=(*batch_shape, n_sample)
    ).reshape(*batch_shape, n_sample, n_var)
    if isinstance(X, torch.Tensor):
        noises = torch.tensor(noises, dtype=X.dtype, device=X.device)
    samples = X[..., None, :] + noises
    return samples


def add_noise(X, sampling_func, sampling_cfg={}, **kwargs):
    batch_shape = X.shape[:-1]
    event_dim = X.shape[-1]
    noise = sampling_func(**sampling_cfg, size=(*batch_shape, 1) ).reshape(*batch_shape, event_dim)
    if isinstance(X, torch.Tensor):
        noise = torch.tensor(noise, dtype=X.dtype, device=X.device)
    return X + noise


class AdditionalFeatures(trans.input.AppendFeatures):
    def transform(self, X):
        expanded_features = self._f(X[..., self.indices], **self.fkwargs)
        return X, expanded_features, torch.zeros(size=(*X.shape[:-1], 0), dtype=X.dtype,
                                                 device=X.device)  # xc, xc_std/samples, xe


class TransformFeature(trans.input.AppendFeatures):
    def transform(self, X):
        transformed_features = self._f(X[..., self.indices], **self.fkwargs)
        return transformed_features, torch.zeros(size=(*X.shape[:-1], 0), dtype=X.dtype,
                                                 device=X.device)  # xc, xc_std/samples, xe


class SelectMultiInputs(trans.input.InputTransform, torch.nn.Module):
    def __init__(
            self,
            sel_indices,
            transform_on_train: bool = True,
            transform_on_eval: bool = True,
            transform_on_fantasize: bool = True,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.register_buffer("sel_indices", sel_indices)

    def transform(self, X):
        return tuple(X[i] for i in self.sel_indices)


class MultiInputTransform(trans.input.InputTransform, ModuleDict):
    r"""An input transform representing the chaining of individual transforms."""

    def __init__(self, **transforms) -> None:
        r"""Chaining of input transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.

        """
        super().__init__(OrderedDict(transforms))
        self.transform_on_train = False
        self.transform_on_eval = False
        self.transform_on_fantasize = False
        for tf in transforms.values():
            self.is_one_to_many |= tf.is_one_to_many
            self.transform_on_train |= tf.transform_on_train
            self.transform_on_eval |= tf.transform_on_eval
            self.transform_on_fantasize |= tf.transform_on_fantasize

    def transform(self, X):
        ret = tuple([tf.transform(X[ind]) for ind, tf in enumerate(self.values())])
        return ret

    def untransform(self, X):
        ret = tuple([tf.untransform(X[ind]) for ind, tf in enumerate(self.values())])
        return ret

    def equals(self, other: trans.input.InputTransform) -> bool:
        return super().equals(other=other) and all(
            t1.equals(t2) for t1, t2 in zip(self.values(), other.values())
        )

    def preprocess_transform(self, X):
        ret = tuple([tf.preprocess_transform(X[ind]) for ind, tf in enumerate(self.values())])
        return ret


class DummyTransform(trans.input.ReversibleInputTransform, torch.nn.Module):
    def __init__(
            self,
            transform_on_train: bool = True,
            transform_on_eval: bool = True,
            transform_on_fantasize: bool = True,
            reverse: bool = False,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    def _transform(self, X: Tensor) -> Tensor:
        return X

    def _untransform(self, X: Tensor) -> Tensor:
        return X


class ScaleTransform(trans.input.Normalize):
    def _transform(self, X: Tensor) -> Tensor:
        return X / self.coefficient

    def _untransform(self, X: Tensor) -> Tensor:
        return X * self.coefficient


if __name__ == '__main__':
    from functools import partial
    from scipy import stats

    xc_sample_size = 1000
    n_var = 1
    raw_input_std = 1.0
    input_distrib = stats.norm(loc=0, scale=raw_input_std)
    input_sampling_func = partial(input_distrib.rvs)

    x = torch.linspace(0, 100, 10).view(-1, 1)
    it_mmd = trans.input.ChainedInputTransform(
        tf1=AdditionalFeatures(f=additional_xc_samples, transform_on_train=True,
                               fkwargs={'n_sample': xc_sample_size, 'n_var': n_var,
                                        'sampling_func': input_sampling_func, }
                               ),
        tf2=MultiInputTransform(
            tf1=trans.input.Normalize(d=n_var, bounds=torch.tensor([[0], [100]])),
            tf2=trans.input.Normalize(d=n_var, bounds=torch.tensor([[0], [100]])),
            tf3=DummyTransform()
        ),
    )
    x_mmd = it_mmd.transform(x)

    it_skl = trans.input.ChainedInputTransform(
        tf1=AdditionalFeatures(f=additional_std, transform_on_train=True,
                               fkwargs={'std': raw_input_std}
                               ),
        tf2=MultiInputTransform(
            tf1=trans.input.Normalize(d=n_var, bounds=torch.tensor([[0], [100]])),
            tf2=ScaleTransform(d=n_var, bounds=torch.tensor([[0], [100]])),
            tf3=DummyTransform()
        ),
    )
    x_skl = it_skl.transform(x)
