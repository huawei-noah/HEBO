import re
from itertools import groupby

import numpy as np

from utilities.aa_utils import idx_to_aa

COUNT_AA = 5  # maximum number of consecutive AAs
N_glycosylation_pattern = 'N[^P][ST][^P]'


def check_constraint_satisfaction(x):
    # Constraints on CDR3 sequence

    aa_seq = ''.join(idx_to_aa[int(aa)] for aa in x)

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
    constraints_satisfied = list(map(lambda seq: check_constraint_satisfaction(seq), x))
    return np.array(constraints_satisfied)