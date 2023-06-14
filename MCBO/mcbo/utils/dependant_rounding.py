""" :func:`DepRound`: implementation of the dependent rounding procedure, from [[Dependent rounding and its applications to approximation algorithms, by R Gandhi, S Khuller, S Parthasarathy, Journal of the ACM, 2006](http://dl.acm.org/citation.cfm?id=1147956)].
It solves the problem of efficiently selecting a set of :math:`k` distinct actions from :math:`\{1,\dots,K\}`, while satisfying the condition that each action :math:`i` is selected with probability :math:`p_i` exactly.
The distribution :math:`(p_1, \dots, p_K)` on :math:`\{1,\dots,K\}` is assumed to be given.
Dependent rounding developed by [Gandhi et al.] is a kind of technique that randomly selects a set of edges from a bipartite graph under some cardinality constraints.

- It runs in :math:`\mathcal{O}(K)` space complexity, and at most :math:`\mathcal{O}(K^2)` time complexity (note that the article [Uchiya et al., 2010] wrongly claim it is in :math:`\mathcal{O}(K)`).
- References: see also https://www.cs.umd.edu/~samir/grant/jacm06.pdf
"""

# Implementation taken from https://github.com/98k-bot/SMPyBandits/blob/35e675bde29dafbec68288fcfcd14ef3b0f058b2/PoliciesMultiPlayers/DepRound.py and https://github.com/98k-bot/SMPyBandits/blob/35e675bde29dafbec68288fcfcd14ef3b0f058b2/Policies/with_proba.py

from __future__ import division, print_function  # Python 2 compatibility

__author__ = ""
__version__ = ""

from random import random

import numpy as np


# --- Utility functions
def with_proba(epsilon):
    """Bernoulli test, with probability :math:`\varepsilon`, return `True`, and with probability :math:`1 - \varepsilon`, return `False`.

    Example:

    >>> from random import seed; seed(0)  # reproductible
    >>> with_proba(0.5)
    False
    >>> with_proba(0.9)
    True
    >>> with_proba(0.1)
    False
    >>> if with_proba(0.2):
    ...     print("This happens 20% of the time.")
    """
    assert 0 <= epsilon <= 1, "Error: for 'with_proba(epsilon)', epsilon = {:.3g} has to be between 0 and 1 to be a valid probability.".format(
        epsilon)  # DEBUG
    return random() < epsilon  # True with proba epsilon


# --- Utility functions
def DepRound(weights_p, k=1, isWeights=True):
    r""" [[Algorithms for adversarial bandit problems with multiple plays,
    by T.Uchiya, A.Nakamura and M.Kudo, 2010](http://hdl.handle.net/2115/47057)] Figure 5 (page 15)
    is a very clean presentation of the algorithm.

    Args:
        weights_p:  :math:`k < K` and weights_p :math:`= (p_1, \dots, p_K)`
                    such that :math:`\sum_{i=1}^{K} p_i = k` (or :math:`= 1`).

    Returns:
        A subset of :math:`\{1,\dots,K\}` with exactly :math:`k` elements.
        Each action :math:`i` is selected with probability exactly :math:`p_i`.

    Example:

    >>> import numpy as np; import random
    >>> np.random.seed(0); random.seed(0)  # for reproductibility!
    >>> K = 5
    >>> k = 2

    >>> weights_p = [ 2, 2, 2, 2, 2 ]  # all equal weights
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [0, 1]

    >>> weights_p = [ 10, 8, 6, 4, 2 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [1, 2]
    >>> DepRound(weights_p, k)
    [3, 4]

    >>> weights_p = [ 3, 3, 0, 0, 3 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 1]

    - See [[Gandhi et al, 2006](http://dl.acm.org/citation.cfm?id=1147956)] for the details.
    """
    p = np.array(weights_p)
    K = len(p)
    # Checks
    assert k < K, "Error: k = {} should be < K = {}.".format(k, K)  # DEBUG
    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(
        p <= 1), "Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ...".format(p)  # DEBUG
    assert np.isclose(np.sum(p), 1), "Error: the sum of weights p_1 + ... + p_K should be = 1 (= {}).".format(
        np.sum(p))  # DEBUG
    # Main loop
    possible_ij = [a for a in range(K) if 0 < p[a] < 1]
    while possible_ij:
        # Choose distinct i, j with 0 < p_i, p_j < 1
        if len(possible_ij) == 1:
            i = np.random.choice(possible_ij, size=1)
            j = i
        else:
            i, j = np.random.choice(possible_ij, size=2, replace=False)
        pi, pj = p[i], p[j]
        assert 0 < pi < 1, "Error: pi = {} (with i = {}) is not 0 < pi < 1.".format(pi, i)  # DEBUG
        assert 0 < pj < 1, "Error: pj = {} (with j = {}) is not 0 < pj < 1.".format(pj, i)  # DEBUG
        assert i != j, "Error: i = {} is different than with j = {}.".format(i, j)  # DEBUG

        # Set alpha, beta
        alpha, beta = min(1 - pi, pj), min(pi, 1 - pj)
        proba = alpha / (alpha + beta)
        if with_proba(proba):  # with probability = proba = alpha/(alpha+beta)
            pi, pj = pi + alpha, pj - alpha
        else:  # with probability = 1 - proba = beta/(alpha+beta)
            pi, pj = pi - beta, pj + beta

        # Store
        p[i], p[j] = pi, pj
        # And update
        possible_ij = [a for a in range(K) if 0 < p[a] < 1]
        if len([a for a in range(K) if np.isclose(p[a], 0)]) == K - k:
            break
    # Final step
    subset = [a for a in range(K) if np.isclose(p[a], 1)]
    if len(subset) < k:
        subset = [a for a in range(K) if not np.isclose(p[a], 0)]
    assert len(
        subset) == k, "Error: DepRound({}, {}) is supposed to return a set of size {}, but {} has size {}...".format(
        weights_p, k, k, subset, len(subset))  # DEBUG
    return subset
