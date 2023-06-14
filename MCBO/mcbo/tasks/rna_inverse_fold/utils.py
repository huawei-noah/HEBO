from ...tasks.data.rna_fold.eterna_data_loader import load_eterna_data
import numpy as np


def get_target_from_id(target_id) -> str:
    eterna_data = load_eterna_data()
    return eterna_data[eterna_data.id == target_id].strc.values[0]


RNA_BASES = np.array(['A', 'C', 'G', 'U'])
