import numpy as np
from bo.base import TestFunction
from task.tools import Absolut, Manual
import torch

class BOTask(TestFunction):
    """
    BO Task Class
    """
    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'
    def __init__(self,
                 device,
                 n_categories,
                 seq_len,
                 bbox=None,
                 normalise=True,
                 mean=None,
                 std=None):
        super(BOTask, self).__init__(normalise)
        self.device = device
        self.bbox = bbox
        self.n_vertices = n_categories
        self.config = self.n_vertices
        self.dim = seq_len
        self.categorical_dims = np.arange(self.dim)
        if self.bbox['tool'] == 'Absolut':
            self.fbox = Absolut(self.bbox)
        elif self.bbox['tool'] == 'manual':
            self.fbox = Manual(self.bbox)
        else:
            assert 0,f"{self.bbox['tool']} Not Implemented"


    def compute(self, x):
        '''
        x: categorical vector
        '''
        energy, _ = self.fbox.Energy(x)
        energy = torch.tensor(energy, dtype=torch.float32).to(self.device)
        return energy

    def idx_to_seq(self, x):
        seqs = []
        for seq in x:
            seqs.append(''.join(self.fbox.idx_to_AA[int(aa)] for aa in seq))
        return seqs