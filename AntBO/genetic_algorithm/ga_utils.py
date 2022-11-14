import yaml

import numpy as np

from itertools import groupby
import re

# from Bio.SeqUtils.ProtParam import ProteinAnalysis

AAs = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
idx_to_AA = {i: aa for aa, i in AA_to_idx.items()}
COUNT_AA = 5  # maximum number of consecutive AAs
N_glycosylation_pattern = 'N[^P][ST][^P]'


def sample_to_aa_seq(x):
    return "".join(idx_to_AA[idx] for idx in x.astype(int))


def check_constraint_satisfaction(x):
    # Constraints on CDR3 sequence

    aa_seq = ''.join(idx_to_AA[int(aa)] for aa in x)

    # Charge of AA
    # prot = ProteinAnalysis(aa_seq)
    # charge = prot.charge_at_pH(7.4)

    # check constraint 1 : charge between -2.0 and 2.0
    charge = 0
    for char in aa_seq:
        charge += int(char == 'R' or char == 'K') + 0.1 * int(char == 'H') - int(char == 'D' or char == 'E')

    if (charge > 2.0 or charge < -2.0):
        return False

    # check constraint 2 : does not contain N-X-S/T pattern. This looks for the single letter code N, followed by any
    # character that is not P, followed by either an S or a T, followed by any character that is not a P. Source
    # https://towardsdatascience.com/using-regular-expression-in-genetics-with-python-175e2b9395c2
    if re.search(N_glycosylation_pattern, aa_seq):
        return False

    # check constraint 3 : any amino acid should not repeat more than 5 times
    # Maximum number of the same subsequent AAs
    count = max([sum(1 for _ in group) for _, group in groupby(x)])
    if (count > COUNT_AA):
        return False

    # # Check the instability index
    # prot = ProteinAnalysis(aa_seq)
    # instability = prot.instability_index()
    # if (instability > 40):
    #     return False

    return True


def check_constraint_satisfaction_batch(x):
    constraints_satisfied = np.zeros(len(x))

    for i in range(len(x)):
        constraints_satisfied[i] = check_constraint_satisfaction(x[i])

    return constraints_satisfied


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, save_path):
    with open(save_path, 'w') as file:
        documents = yaml.dump(config, file)
