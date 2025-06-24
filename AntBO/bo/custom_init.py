import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent)
INIT_DATA_PATH = os.path.join(ROOT_PROJECT, 'init_dataset')


def get_n_per_cat(n_loosers: int, n_mascottes: int, n_heroes):
    return dict(Loosers=n_loosers, Mascotte=n_mascottes, Heroes=n_heroes)


def get_top_cut_ratio_per_cat(top_cut_ratio_loosers: int, top_cut_ratio_mascottes: int, top_cut_ratio_heroes):
    return dict(Loosers=top_cut_ratio_loosers, Mascotte=top_cut_ratio_mascottes, Heroes=top_cut_ratio_heroes)


class InitialBODataset:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_categories(self) -> np.ndarray:
        return self.data['Type'].values

    def get_index_encoded_x(self) -> np.ndarray:
        return np.vstack(self.data['AA to ind'].values)

    def get_protein_names(self) -> pd.Series:
        return self.data['Protein']

    def get_protein_binding_energy(self) -> pd.Series:
        return self.data['Binding Energy']

    def __len__(self) -> int:
        return len(self.data)


def get_initial_dataset_path(antigen_name: str, n_per_cat: Dict[str, int], top_cut_ratio_per_cat: Dict[str, float],
                             seed: int) -> str:
    """

    Parameters
    ----------
    antigen_name: name of the antigen
    n_per_cat: dictionary {category: number_of_samples}
    top_cut_ratio_per_cat: dictionary {category: top_cut_ratio}
    seed: seed used to generate the dataset

    Returns
    -------

    """
    init_dataset_root = os.path.join(INIT_DATA_PATH, antigen_name, str(seed))
    init_dataset_id: str = ""
    for cat, n_sample in n_per_cat.items():
        if n_sample > 0:
            init_dataset_id += f"{cat}-{n_sample:d}_"
    for cat, top_cut_ratio in top_cut_ratio_per_cat.items():
        if top_cut_ratio > 0:
            init_dataset_id += f"{cat}-{top_cut_ratio:g}_"
    init_dataset_id = init_dataset_id[:-1]
    init_dataset_folder_path = os.path.join(init_dataset_root, init_dataset_id, "init_data")
    os.makedirs(os.path.dirname(init_dataset_folder_path), exist_ok=True)
    return init_dataset_folder_path


def get_initial_dataset_path_(antigen_name: str, top_category: str, n_samples: int, top_cat_top_cut_ratio: float,
                              seed: int) -> str:
    """

    Parameters
    ----------
    antigen_name: name of the antigen
    top_category: name of the top category of protein included in the initial dataset
    n_samples: number of samples in this initial dataset
    top_cat_top_cut_ratio:
    seed

    Returns
    -------

    """
    init_dataset_root = os.path.join(INIT_DATA_PATH, antigen_name, str(seed))
    init_dataset_id = f"{top_category}_n-{n_samples}_top-cat-cut-{top_cat_top_cut_ratio:g}"
    init_dataset_folder_path = os.path.join(init_dataset_root, init_dataset_id, "init_data")
    os.makedirs(os.path.dirname(init_dataset_folder_path), exist_ok=True)
    return init_dataset_folder_path
