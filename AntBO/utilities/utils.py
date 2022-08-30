AAs = 'ACDEFGHIKLMNPQRSTVWY'
AA_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
idx_to_AA = {i: aa for aa, i in AA_to_idx.items()}


def sample_to_aa_seq(x):
    return "".join(idx_to_AA[idx] for idx in x.astype(int))