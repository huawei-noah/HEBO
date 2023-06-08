# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import torch
import pickle
import numpy as np

from nap.RL.util import compute_cond_gps
from nap.environment.hpo import get_hpo_specs
from nap.environment.objectives import get_HPO_domain
from pathlib import Path


if __name__ == '__main__':
    rootdir = os.path.join(os.path.dirname(Path(os.path.realpath(__file__)).parent))
    hpo_type = "hpobenchXGB"
    dims, points, train_datasets, valid_datasets, test_datasets, kernel_lengthscale, kernel_variance, \
        noise_variance, X_mean, X_std = get_hpo_specs(hpo_type, rootdir)

    saved_models_dir = os.path.join("/".join(train_datasets[0].split("/")[:-1]), 'GPs/train_sets')
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)

    loaded_datasets = [pickle.load(open(dataset, "rb")) for dataset in train_datasets]
    all_X = np.array([get_HPO_domain(data=dataset) for dataset in loaded_datasets])
    all_X = all_X.reshape(-1, all_X.shape[-1])
    compute_cond_gps(train_datasets, saved_models_dir, trainXmean=all_X.mean(0), trainXstd=all_X.std(0))
