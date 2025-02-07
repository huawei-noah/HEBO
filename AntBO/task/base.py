from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
from matplotlib.axes import Axes


class BaseTask(ABC):
    def __init__(self) -> None:
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}

    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        x: categorical vector
        """
        raise NotImplementedError

    def plot_energy(self, x: np.ndarray) -> Axes:
        """
        x: (seeds x trials) numpy array of energy
        """
        raise NotImplementedError

    def visualise_binding(self, x: list[str], y: Optional[str] = None):
        """
        x: CDR3 sequence to visualise
        y: Antibody identifier
        antibody:
        """
        raise NotImplementedError


class BaseTool(ABC):
    def __init__(self) -> None:
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(aa)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}

    @staticmethod
    def convert_array(arr: np.ndarray, conversion_dic: Dict[Any, Any], end_type: Optional) -> np.ndarray:
        new_arr = np.copy(arr).astype(object)
        for k, v in conversion_dic.items():
            new_arr[new_arr == k] = v
        if end_type is not None:
            new_arr = new_arr.astype(end_type)
        return new_arr

    def convert_array_idx_to_aas(self, idx: np.ndarray) -> np.ndarray:
        return self.convert_array(arr=idx, conversion_dic=self.idx_to_AA, end_type=None)

    def convert_array_aas_to_idx(self, aas: np.ndarray) -> np.ndarray:
        return self.convert_array(arr=aas, conversion_dic=self.AA_to_idx, end_type=int)

    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        x: categorical vector
        """
        raise NotImplementedError
