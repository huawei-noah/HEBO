# 2021.11.10-Add management of saving folders
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

import os
import pickle
import time
from pathlib import Path
from typing import Any, List, Optional

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
NAME = os.getcwd().split('/')[2]
PROJECT_NAME = os.path.basename(ROOT_PROJECT)

_aux_path = os.path.join(ROOT_PROJECT, "utils", 'results_storage_root_path.txt')
try:
    f_ = open(_aux_path, "r")
    DATA_STORAGE_ROOT = f_.readlines()[0]
    if DATA_STORAGE_ROOT[-1] == '\n':
        DATA_STORAGE_ROOT = DATA_STORAGE_ROOT[:-1]
    f_.close()
except FileNotFoundError as e_:
    aux = '/my/absolute/path/'
    raise FileNotFoundError(f'File \n - {_aux_path}\n'
                            f'not found,\n'
                            f'   --> shall create it and fill it with one line describing the '
                            f'root path where you want all results to be stored, e.g. running\n'
                            f"\techo '{aux}' > {_aux_path}\n"
                            f'in which case your results, data and saved models will be stored in:\n'
                            f'\t{aux}{PROJECT_NAME}/') from e_


def get_storage_root():
    return os.path.join(DATA_STORAGE_ROOT, NAME, PROJECT_NAME, 'results')


def get_storage_models_root():
    return os.path.join(DATA_STORAGE_ROOT, NAME, PROJECT_NAME, 'models')


def get_storage_tuning_root():
    return os.path.join(get_storage_root(), 'tuning')


def get_storage_data_root():
    return os.path.join(DATA_STORAGE_ROOT, NAME, PROJECT_NAME, 'data')


def get_storage_datasets_root():
    return os.path.join(get_storage_data_root(), 'datasets')


def get_mtm_data_root():
    return os.path.join(DATA_STORAGE_ROOT, NAME, PROJECT_NAME, 'data/epfl-benchmark/mtm/')


def str_dict(d):
    """ """
    s = []
    for k, v in d.items():
        if isinstance(v, dict):
            str_val = str_dict(v)
        elif isinstance(v, list):
            str_val = str_list(v)
        else:
            str_val = str(v)
        s.extend([k, str_val.replace('/', '~')])
    return '-'.join(s)


def str_list(l: List):
    return '_'.join(map(str, l))


def save_w_pickle(obj: Any, path: str, filename: Optional[str] = None) -> None:
    """ Save object obj in file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_w_pickle(path: str, filename: Optional[str] = None) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError as e:
            raise


def safe_load_w_pickle(path: str, filename: Optional[str] = None, n_trials=3, time_sleep=2) -> Any:
    """ Make several attempts to load object from file exp_path/filename.pkl """
    trial = 0
    end = False
    result = None
    while not end:
        try:
            result = load_w_pickle(path=path, filename=filename)
            end = True
        except (pickle.UnpicklingError, EOFError) as e:
            trial += 1
            if trial > n_trials:
                raise e
            time.sleep(time_sleep)
    return result


if __name__ == '__main__':
    print(f"Results will be stored in {get_storage_root()}")
    print(f"Data should be stored in {get_storage_data_root()}")
