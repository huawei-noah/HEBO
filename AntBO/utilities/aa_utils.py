# Possible amino acids
aas = 'ACDEFGHIKLMNPQRSTVWY'

# Mapping from amino acids to indices
aa_to_idx = {aa: idx for idx, aa in enumerate(aas)}

# Mapping from indices to amino acids
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}

# Possible indices
valid_aa_indices = [idx for idx in range(len(aas))]

# Convert indices to amino acid sequence
def indices_to_aa_seq(x):
    return "".join(idx_to_aa[idx] for idx in x.astype(int))