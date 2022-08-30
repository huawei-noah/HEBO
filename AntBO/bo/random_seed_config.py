# Lifted from COMBO code

import numpy as np


SEED_STR_LIST = ['ISING', 'CONTAMINATION', 'AEROSTRUCTURAL', 'TRAVELPLAN',
                 'PESTCONTROL', 'CENTROID', 'HIGHORDERBINARY', 'MAXSAT']


def generate_random_seed_pair_ising():
    return _generate_random_seed_pair('ISING', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_pair_contamination():
    return _generate_random_seed_pair('CONTAMINATION', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_aerostruct():
    return _generate_random_seed('AEROSTRUCTURAL', n_init_point_seed=10)


def generate_random_seed_pair_travelplan():
    return _generate_random_seed_pair('TRAVELPLAN', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_pestcontrol():
    return _generate_random_seed('PESTCONTROL', n_init_point_seed=25)


def generate_random_seed_pair_centroid():
    return _generate_random_seed_pair('CENTROID', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_maxsat():
    return _generate_random_seed('MAXSAT', n_init_point_seed=25)


def _generate_random_seed(seed_str, n_init_point_seed=10):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))


def _generate_random_seed_pair(seed_str, n_test_case_seed=5, n_init_point_seed=5):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_test_case_seed):
        result[rng_state.randint(0, 10000)] = list(rng_state.randint(0, 10000, (n_init_point_seed, )))
    return result


if __name__ == '__main__':
    print(generate_random_seed_maxsat())