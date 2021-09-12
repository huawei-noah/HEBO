# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pickle
from pathlib import Path
from typing import Any

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)

_aux_path = os.path.join(ROOT_PROJECT, "utils", 'storage_root_path.txt')
try:
    f_ = open(_aux_path, "r")
    DATA_STORAGE_ROOT = f_.readlines()[0]
    if DATA_STORAGE_ROOT[-1] == '\n':
        DATA_STORAGE_ROOT = DATA_STORAGE_ROOT[:-1]
    f_.close()
except FileNotFoundError as e_:
    aux = '~/LSO-storage/'
    raise FileNotFoundError(f'File {_aux_path} not found, shall create it and fill it with one line describing the '
                            f'root path where you want all results to be stored, e.g. running\n'
                            f"\techo '{aux}' > {_aux_path}\n"
                            f'in which case your results will be stored in:\n'
                            f'\t{aux}') from e_


def str_dict(d):
    """ """
    s = []
    for k, v in d.items():
        s.extend([k, str(v)])
    return '-'.join(s)


def save_w_pickle(obj: Any, path: str, filename: str) -> None:
    """ Save object obj in file exp_path/filename.pkl """
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_w_pickle(path: str, filename: str) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError as e:
            print(path, filename)
            raise


def get_storage_root():
    return os.path.join(DATA_STORAGE_ROOT)


def get_storage_results_root():
    return os.path.join(get_storage_root(), 'results')


def get_data_root():
    return os.path.join(get_storage_root(), 'data')
