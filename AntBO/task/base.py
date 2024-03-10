from abc import ABC
from typing import Any, Dict, Optional

import numpy as np


class BaseTask(ABC):
    def __init__(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(AA)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}

    def energy(self, x):
        '''
        x: categorical vector
        '''
        raise NotImplementedError

    def plotEnergy(self, x):
        '''
        x: (seeds x trials) numpy array of energy
        '''
        raise NotImplementedError

    def visualiseBinding(self, x, y=None):
        '''
        x: CDR3 sequence to visualise
        y: Antibody identifier
        antibody:
        '''
        raise NotImplementedError


class BaseTool(ABC):
    def __init__(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(AA)}
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

    def Energy(self, x):
        '''
        x: categorical vector
        '''
        raise NotImplementedError
