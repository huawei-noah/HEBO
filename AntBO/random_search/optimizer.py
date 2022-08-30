import numpy as np

from utilities.aa_utils import valid_aa_indices
from utilities.constraint_utils import check_constraint_satisfaction_batch


class Optimizer:
    def __init__(self, config):
        self.seq_len = config['sequence_length']
        self.aa_index_bounds = [valid_aa_indices[0], valid_aa_indices[-1]]

    def suggest(self, n_suggestions=1):
        # Create the initial population. Last column stores the fitness
        X = np.random.randint(low=self.aa_index_bounds[0], high=self.aa_index_bounds[1] + 1,
                              size=(n_suggestions, self.seq_len))

        # Check for constraint violation
        invalid_indices = np.logical_not(check_constraint_satisfaction_batch(X))

        # Continue until all samples satisfy the constraints
        while np.sum(invalid_indices) != 0:
            # Generate new samples for the ones that violate the constraints
            X[invalid_indices] = np.random.randint(low=self.aa_index_bounds[0], high=self.aa_index_bounds[1] + 1,
                              size=(np.sum(invalid_indices), self.seq_len))

            # Check for constraint violation
            invalid_indices = np.logical_not(check_constraint_satisfaction_batch(X))

        return X

    def observe(self, X, fX):
        pass
