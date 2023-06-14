from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd


class SfuFunction(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __init__(self, num_dims: Union[int, List[int]],
                 lb: Union[float, np.ndarray],
                 ub: Union[float, np.ndarray]):
        assert isinstance(num_dims, (int, list)), num_dims
        assert isinstance(lb, int) or isinstance(lb, float) or isinstance(lb, np.ndarray)
        assert isinstance(ub, int) or isinstance(ub, float) or isinstance(ub, np.ndarray)

        self.num_dims = num_dims
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        pass
