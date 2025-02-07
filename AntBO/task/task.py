import numpy as np

from task.base import BaseTask
from task.utils import plot_mean_std
from task.tools import Absolut, Visualisation

class Task(BaseTask):
    def __init__(self,
                 config):
        BaseTask.__init__(self)
        '''
        config: dictionary of parameters
                bbox: 'Absolut', etc., Tool for computing binding energy
                antigen: PDB ID of antigen
                Absolut: dictionary of Absolut parameters
                docktool: 'ClusPro', etc., Tool to use for docking
                vistool: 'PyMol', etc., Tool to use for visualisation
        '''
        self.config = config
        self.Vis = Visualisation(self.config['antigen'],
                                               self.config['docktool'],
                                               self.config['vistool'])

        if self.config['bbox'] == 'Absolut':
            self.Binding = Absolut(self.config['Absolut'])
        else:
            assert 0,f"{self.config['method']} Not Implemented"

    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        '''
        x: categorical vector
        '''
        return self.Binding.energy(x)

    def plot_energy(self, x):
        '''
        x: (seeds x trials) numpy array of energy

        Returns:
            Axis of Plot
        '''
        return plot_mean_std(x)

    def visualise_binding(self, x, y=None):
        '''
        x: CDR3 sequence to visualise
        y: Antibody identifier
        antibody:
        '''
        raise self.Vis.visualise(x, y)