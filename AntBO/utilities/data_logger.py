import os
import numpy as np
import pandas as pd


class DataLogger:

    def __init__(self, size):
        self.size = size
        self._idx = 0
        self.df = self.res = pd.DataFrame(np.nan, index=np.arange(1, self.size + 1),
                                          columns=['Num BB Evals', 'Last Binding Energy', 'Best Binding Energy',
                                                   'Suggest Time', 'Last Protein', 'Best Protein'])

    def append(self, protein, binding_energy, suggest_time, num_bb_evals):

        if self._idx == 0:
            best_binding_energy = binding_energy
            best_protein = protein

        else:
            best_idx = self.df.iloc[:self._idx]['Last Binding Energy'].argmin()
            best_binding_energy = self.df.iloc[best_idx]['Last Binding Energy']
            best_protein = self.df.iloc[best_idx]['Last Protein']

            if best_binding_energy > binding_energy:
                best_binding_energy = binding_energy
                best_protein = protein

        self.df.iloc[self._idx] = [int(num_bb_evals), binding_energy, best_binding_energy, suggest_time, protein,
                                   best_protein]

        self._idx += 1

    def append_batch(self, proteins, binding_energies, suggest_times, num_bb_evals):

        for protein_, binding_energy_, suggest_time_, num_bb_evals_ in zip(proteins, binding_energies, suggest_times,
                                                                           num_bb_evals):
            self.append(protein_, binding_energy_, suggest_time_, num_bb_evals_)

    def save_results(self, save_dir):
        self.df.to_csv(os.path.join(save_dir, 'results.csv'))

    def reset_logger(self):
        self._idx = 0
        self.df = self.res = pd.DataFrame(np.nan, index=np.arange(1, self.size + 1),
                                          columns=['Num BB Evals', 'Last Binding Energy', 'Best Binding Energy',
                                                   'Suggest Time', 'Last Protein', 'Best Protein'])


if __name__ == '__main__':

    import random
    import string

    n = 20

    logger = DataLogger(n)

    for i in range(n):
        logger.append(''.join(random.choice(string.ascii_uppercase) for _ in range(10)), np.random.randn(),
                      np.random.random(), i + 1)
