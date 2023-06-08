# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import pickle
import os

import numpy as np

from nap.RL.util import print_best_gp_params
from nap.environment.objectives import get_HPO_domain


def get_hpo_specs(hpo_type, root_dir=None):
    dims = {"hpobenchXGB": 6, "glmnet": 2, "rpart_preproc": 3, "xgboost": 16, 'rpart': 6, "ranger": 6, 'svm': 8}
    points = {"hpobenchXGB": 1000, "glmnet": 100, "rpart_preproc": 300, "xgboost": 50, 'rpart': 200, "ranger": 50, 'svm': 300}
    limits = {"hpobenchXGB": range(20), "rpart_preproc": range(36), "glmnet": [idx for idx in range(27) if idx not in [6]],
              "xgboost": range(24), "ranger": range(20), 'svm': [idx for idx in range(51) if idx not in [0,1]],
              'rpart': [idx for idx in range(56) if idx not in [0,1,5,23,24,30,47]]}
    limits_valid = {"hpobenchXGB": range(20, 34), "rpart_preproc": range(4), "glmnet": range(3), "xgboost": range(3),
                    "ranger": range(2), 'svm': range(6), 'rpart': range(7)}
    limits_test = {"hpobenchXGB": range(34, 48), "rpart_preproc": range(4), "glmnet": range(3), "xgboost": range(2),
                   "ranger": range(2), 'svm': range(6), 'rpart': range(6)}
    num_dims = {"hpobenchXGB": list(range(6)), "glmnet": [0, 1], "rpart_preproc": [0, 1, 2],
                "xgboost": [0, 1, 2, 3, 4, 5, 6, 7, 8], "rpart": [0, 1, 2, 3], "ranger": [0, 1, 2], "svm": [0, 1, 2]}
    cat_dims = {"hpobenchXGB": [], "glmnet": [], "rpart_preproc": [], "xgboost": [9, 10, 11, 12, 13, 14],
                "rpart": [4, 5], "ranger": [3, 4], "svm": [3, 4, 5]}
    num_classes = {"hpobenchXGB": [], "glmnet": [], "rpart_preproc": [], "xgboost": [2, 2, 2, 2, 2, 2],
                   "rpart": [2, 2], "ranger": [2, 2], "svm": [2, 2, 3]}
    cat_alphabet = {k: {cd: list(range(nc)) for cd, nc in zip(cat_dims[k], num_classes[k])} for k in num_classes}
    na_dims = {"hpobenchXGB": {}, "glmnet": {}, "rpart_preproc": {}, "xgboost": {9: 1, 10: 2, 11: 4, 12: 5, 13: 8},
               "rpart": {4: 0, 5: 3}, "ranger": {3: 0}, "svm": {3: 1, 4: 2}}
    nom_dims = {"hpobenchXGB": [], "glmnet": [], "rpart_preproc": [], "xgboost": [14], "rpart": [], "ranger": [4], "svm": [5]}

    kernel_lengthscale = {
        "hpobenchXGB": [8.10816999, 3.41942592, 0.46064765, 65.46127277, 39.59250354, 40.04858487],
    }
    kernel_variance = {"hpobenchXGB": 0.10367898}
    noise_variance = {"hpobenchXGB": 0.0003625}

    if hpo_type in ["rpart_preproc", "glmnet", "xgboost", "ranger", "svm", "rpart"]:
        if root_dir is not None:
            root_dir = f"{root_dir}/HPOB_data"
        else:
            root_dir = "HPOB_data"
        train_datasets = [f"{root_dir}/{hpo_type}_train_{i}.pkl" for i in limits[hpo_type]]
        valid_datasets = [f"{root_dir}/{hpo_type}_validation_{i}.pkl" for i in limits_valid[hpo_type]]
        test_datasets = [f"{root_dir}/{hpo_type}_test_{i}.pkl" for i in limits_test[hpo_type]]
    else:
        if root_dir is not None:
            path_func = lambda type, i: os.path.join(root_dir, f"HPO_data/{hpo_type}_{i}_eq.pkl")
        else:
            path_func = lambda type, i: f"HPO_data/{hpo_type}_{i}_eq.pkl"
        train_datasets = [path_func(hpo_type, i) for i in limits[hpo_type]]
        valid_datasets = [path_func(hpo_type, i) for i in limits_valid[hpo_type]]
        test_datasets = [path_func(hpo_type, i) for i in limits_test[hpo_type]]

    loaded_datasets = [pickle.load(open(dataset, "rb")) for dataset in train_datasets]
    all_X = np.concatenate([get_HPO_domain(data=dataset) for dataset in loaded_datasets], 0)

    # # uncomment below to get GP params
    # loaded_datasets = [pickle.load(open(dataset, "rb")) for dataset in train_datasets + valid_datasets]
    # all_X2 = np.array([get_HPO_domain(data=dataset) for dataset in loaded_datasets])
    # all_Y2 = np.array([dataset['accs'] for dataset in loaded_datasets])
    # print_best_gp_params(all_X2, all_Y2)

    return dims[hpo_type], num_dims[hpo_type], cat_dims[hpo_type], cat_alphabet[hpo_type], na_dims[hpo_type], nom_dims[hpo_type], \
           points[hpo_type], train_datasets, valid_datasets, test_datasets, \
           kernel_lengthscale[hpo_type] if hpo_type in kernel_lengthscale else None, \
           kernel_variance[hpo_type] if hpo_type in kernel_variance else None, \
           noise_variance[hpo_type] if hpo_type in noise_variance else None, \
           all_X.mean(0).tolist(), all_X.std(0).tolist()


def get_cond_hpo_specs(hpo_type, root_dir=None):
    limits = {"hpobenchXGB": range(20), "rpart_preproc": range(36), "glmnet": [idx for idx in range(27) if idx not in [6]],
              "xgboost": range(24), "ranger": range(20), 'svm': [idx for idx in range(51) if idx not in [0, 1]],
              'rpart': [idx for idx in range(56) if idx not in [0, 1, 5, 23, 24, 30, 47]]}

    dims, num_dims, cat_dims, cat_alphabet, na_dims, nom_dims, points, \
    train_datasets, valid_datasets, test_datasets, \
    kernel_lengthscale, kernel_variance, noise_variance, \
    X_mean, X_std = get_hpo_specs(hpo_type, root_dir)

    if hpo_type in ["rpart_preproc", "glmnet", "xgboost", "ranger", "svm", "rpart"]:
        if root_dir is not None:
            root_dir = f"{root_dir}/HPOB_data"
        else:
            root_dir = "HPOB_data"
        models_path_func = lambda type, i: os.path.join(root_dir, f"gps/{hpo_type}_train_{i}_gp.pt")
        train_datasets = [models_path_func(hpo_type, i) for i in limits[hpo_type]]

    else:
        if root_dir is not None:
            root_dir = f"{root_dir}/HPO_data"
        else:
            root_dir = "HPO_data"
        models_path_func = lambda type, i: os.path.join(root_dir, f"GPs/train_sets/{hpo_type}_{i}_eq_gp_model.pt")
        train_datasets = [models_path_func(hpo_type, i) for i in limits[hpo_type]]

    return dims, num_dims, cat_dims, cat_alphabet, na_dims, nom_dims, points, train_datasets, valid_datasets, test_datasets, \
           kernel_lengthscale, kernel_variance, noise_variance
