import numpy as np
import os
from abc import ABC

class BaseModel(ABC):
    def __init__(self):
        pass

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError
