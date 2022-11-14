import re
from itertools import groupby

COUNT_AA = 5
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: i for i, aa in enumerate(AA)}
idx_to_AA = {value: key for key, value in AA_to_idx.items()}
N_glycosylation_pattern = 'N[^P][ST][^P]'

def check_cdr_constraints(x):
    # Constraints on CDR3 sequence
    x_to_seq = ''.join(idx_to_AA[int(aa)] for aa in x)
    #prot = ProteinAnalysis(x_to_seq)
    #charge = prot.charge_at_pH(7.4)
    # Counting
    count = max([sum(1 for _ in group) for _, group in groupby(x_to_seq)])
    if count>5:
        return False
    charge = 0
    for char in x_to_seq:
        charge += int(char == 'R' or char == 'K') + 0.1 * int(char == 'H') - int(char == 'D' or char == 'E')
    if (charge > 2.0 or charge < -2.0):
        return False

    if re.search(N_glycosylation_pattern, x_to_seq):
        return False

    #stability = prot.instability_index()
    #if stability>40:
    #    return False
    return True