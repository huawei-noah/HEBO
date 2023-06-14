from typing import List
import numpy as np


def sample_from_simplex(d: int, n_samples: int) -> np.ndarray:
    """
    Sample m points uniformly in the unit (d-1)-simplex
    Use  Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134 method.
    """
    # Sample m, d+1 values in U[0,1]
    sorted_samples = np.random.uniform(0, 1, (n_samples, d + 1))
    # set first elements to 0
    sorted_samples[:, 0] = 0

    # set last elements to 1
    sorted_samples[:, -1] = 1

    # Sort
    sorted_samples.sort(-1)

    # Output the intervals
    return sorted_samples[:, 1:] - sorted_samples[:, :-1]


def diverse_random_dict_sample(m: int, n_cats_per_dim: List[int]) -> np.ndarray:
    """
    Dictionary design for binary input space {0, 1}^d with diversely sparse rows

    Args:
        m: dictionary size
        n_cats_per_dim: number of categories per dimension

    Returns:
        A: array of shape (m, d) corresponding to the basis vectors for dictionary embedding
    """
    a_dict = np.zeros((m, len(n_cats_per_dim)))
    max_n_cat = max(n_cats_per_dim)
    for i in range(m):
        theta = sample_from_simplex(d=max_n_cat, n_samples=1)[0]
        for j in range(len(n_cats_per_dim)):
            if n_cats_per_dim[j] == max_n_cat:
                subthetas = theta
            else:
                subthetas = np.random.choice(a=theta, size=n_cats_per_dim[j], replace=False)
                subthetas /= subthetas.sum()
            a_dict[i, j] = np.random.choice(np.arange(n_cats_per_dim[j]), size=1, p=subthetas)
    return a_dict
