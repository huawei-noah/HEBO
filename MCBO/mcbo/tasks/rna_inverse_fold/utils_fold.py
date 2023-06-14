def rna_fold(rna_seq: str) -> str:
    import RNA
    return RNA.fold(rna_seq)[0]


def get_hamming_distrance(target: str, seq: str) -> int:
    import RNA
    return RNA.hamming_distance(target, seq)
