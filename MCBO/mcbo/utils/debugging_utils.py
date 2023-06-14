# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np


def train_test_split(x, y, train_ratio=0.8):
    n = len(x)
    n_train = int(np.round(train_ratio * n))

    random_indices = np.random.permutation(n)

    x_train = x[random_indices[:n_train]]
    y_train = y[random_indices[:n_train]]
    x_test = x[random_indices[n_train:]]
    y_test = y[random_indices[n_train:]]

    return x_train, y_train, x_test, y_test


def avg_neg_ll(samples, mean, std):
    """ Compute sample-wise negative log-likelihood of samples given mean and std of Gaussian distr. """
    return np.mean(
        np.log(2 * np.pi * (std ** 2)) / 2 + ((samples - mean) ** 2) / (2 * (std ** 2)))


def rmse(samples, mean):
    return np.sqrt(np.mean((samples - mean) ** 2))
