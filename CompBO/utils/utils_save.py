import os
import pickle
from typing import Any
import json

import numpy as np


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




def save_json(obj: Any, path: str, filename: str) -> None:
    """ Save object obj in file exp_path/filename.json """
    if len(filename) < 5 or filename[-5:] != '.json':
        filename += '.json'
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(obj, f)


def load_json(path: str, filename: str) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if len(filename) < 5 or filename[-5:] != '.json':
        filename += '.json'
    with open(os.path.join(path, filename)) as f:
        return json.load(f)


def save_np(array: np.ndarray, path: str, filename: str) -> None:
    """ Save numpy array in file exp_path/filename.npy """
    np.save(os.path.join(path, filename), array)


def load_np(path: str, filename: str) -> np.ndarray:
    """ Load numpy array stored in file exp_path/filename.npy """
    if len(filename) < 5 or filename[-4:] != '.npy':
        filename += '.npy'
    return np.load(os.path.join(path, filename))
