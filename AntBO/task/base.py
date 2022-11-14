from abc import ABC
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

    def Energy(self, x):
        '''
        x: categorical vector
        '''
        raise NotImplementedError
