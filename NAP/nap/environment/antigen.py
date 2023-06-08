## Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os


def get_antigen_datasets(rootdir):
    antigen_data_root = os.path.join(rootdir, 'antigen_data')
    antigen_datasets_paths = os.listdir(antigen_data_root)
    antigen_datasets_paths = [a for a in antigen_datasets_paths if '.csv' in a]
    available_antigens = []
    for antigen in antigen_datasets_paths:
        antigen_gp_path = os.path.join(antigen_data_root, f'{antigen.split(".")[0]}.pt')
        if os.path.exists(antigen_gp_path):
            available_antigens.append(antigen.split(".")[0])

    train_antigens = available_antigens[:int(0.7 * len(available_antigens))]
    val_antigens = available_antigens[int(0.7 * len(available_antigens)):int(0.8 * len(available_antigens))]
    test_antigens = available_antigens[int(0.8 * len(available_antigens)):]

    train_datasets = []
    train_trained_gps = []
    for a in train_antigens:
        antigen_dataset_path = os.path.join(antigen_data_root, f'{a}.csv')
        antigen_gp_path = os.path.join(antigen_data_root, f'{a}.pt')
        train_datasets.append(antigen_dataset_path)
        train_trained_gps.append(antigen_gp_path)

    val_datasets = []
    val_trained_gps = []
    for a in val_antigens:
        antigen_dataset_path = os.path.join(antigen_data_root, f'{a}.csv')
        antigen_gp_path = os.path.join(antigen_data_root, f'{a}.pt')
        val_datasets.append(antigen_dataset_path)
        val_trained_gps.append(antigen_gp_path)

    test_datasets = []
    test_trained_gps = []
    for a in test_antigens:
        antigen_dataset_path = os.path.join(antigen_data_root, f'{a}.csv')
        antigen_gp_path = os.path.join(antigen_data_root, f'{a}.pt')
        test_datasets.append(antigen_dataset_path)
        test_trained_gps.append(antigen_gp_path)
        
    return train_datasets, train_trained_gps, val_datasets, val_trained_gps, test_datasets, test_trained_gps
