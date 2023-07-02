# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch


def remove_repeating_samples(x: torch.Tensor, y: torch.Tensor):
    """
    Function that removes identical samples that have the same y values

    Args:
        x: points in transformed search space
        y: corresponding black-box values (2D-tensor)

    Returns:
        x: points with pairwise distinct y values
        y: corresponding black-box values (2D-tensor)
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]

    num_out = y.shape[1]
    samples = torch.cat((x, y), dim=1)

    # Get all unique samples
    unique, inverse = torch.unique(samples, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    samples = samples[indices]

    # Randomize the order
    random_indices = np.random.permutation(len(samples))
    samples = samples[random_indices]

    x, y = samples[:, :-num_out], samples[:, -num_out:]

    return x, y


def subsample_training_data(x: torch.Tensor, y: torch.Tensor, dataset_size: int) -> (
        torch.FloatTensor, torch.FloatTensor):
    """
    Function used to subsample the training dataset if its larger than dataset_size. This function obtains
    int(dataset_size / 2) of the best points and len(x) - int(dataset_size / 2) random points.

    Args:
        x: input points
        y: input values
        dataset_size: size of dataset subsampled from (x, y)

    Returns:
        x: subsampled input points
        y: subsampled input values
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]

    if len(y) > dataset_size:
        num_best = int(dataset_size / 2)
        num_random = dataset_size - num_best

        best_indices = torch.argsort(y.flatten())
        temp_x = x[best_indices[:num_best]]
        temp_y = y[best_indices[:num_best]]

        # get random indices
        random_indices = np.random.permutation(len(best_indices[num_best:]))
        temp_x = torch.cat((temp_x, x[best_indices[num_best:][random_indices[:num_random]]]), dim=0)
        temp_y = torch.cat((temp_y, y[best_indices[num_best:][random_indices[:num_random]]]), dim=0)

        # randomise the order
        random_indices = np.random.permutation(len(temp_y))
        x = temp_x[random_indices]
        y = temp_y[random_indices]

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]

    return x, y
