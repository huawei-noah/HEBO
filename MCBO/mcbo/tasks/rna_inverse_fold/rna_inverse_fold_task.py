from typing import Union, List, Any, Dict

import numpy as np
import pandas as pd

from ...tasks import TaskBase
from ...tasks.rna_inverse_fold.utils import RNA_BASES
from ...tasks.rna_inverse_fold.utils_fold import rna_fold, get_hamming_distrance


class RNAInverseFoldTask(TaskBase):
    alphabet = RNA_BASES
    valid_target_elements = {".", "(", ")"}
    ordinal_alphabet = np.arange(len(alphabet))

    @property
    def name(self) -> str:
        return 'RNA Inverse Folding'

    def __init__(self, target: str, binary_mode: bool = False):
        """
        Args:
            target: target structure as a string of `(`, `.`, and `)`
            binary_mode: whether categories ACGU should be converted to binary values on 2 bits
        """
        assert len(set(target).difference(self.valid_target_elements)) == 0, set(target).difference(
            self.valid_target_elements)
        assert len(target) > 2, target
        super(RNAInverseFoldTask, self).__init__()
        self.target = target
        self.binary_mode = binary_mode

    @property
    def dim(self):
        if not self.binary_mode:
            return len(self.target)
        else:
            return 2 * len(self.target)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """ Transform entry x into a RNA sequence and evaluate it's fitness given as the Hamming distance
            between the folded RNA sequence and the target
        """

        sequences: List[str] = []
        for i in range(len(x)):
            seq = x.iloc[i].values
            if self.binary_mode:
                rna_seq = self.get_rnaseq_from_bin(seq)
            else:
                if not isinstance(seq[0], str):
                    seq = [self.alphabet[s] for s in np.round(np.array(seq)).astype(int)]
                rna_seq = "".join(seq)
            sequences.append(rna_seq)

        structures = [rna_fold(s) for s in sequences]  # get the folding structure associated to the RNA
        return np.array([get_hamming_distrance(self.target, s) for s in structures]).reshape(-1, 1)

    def get_rnaseq_from_bin(self, x: Union[np.ndarray, List[int]]) -> str:
        """ Convert binary-encoded rna sequence `x` into a proper rna sequence """
        x = np.array(x).astype(int)
        x = x.reshape(2, len(self.target))
        x = x[0] * 2 + x[1]
        return ''.join(self.alphabet[x])

    @staticmethod
    def get_static_search_space_params(binary_mode: bool, target: str) -> List[Dict[str, Any]]:
        if binary_mode:
            params = [{'name': f'x{i + 1}', 'type': 'bool'} for i in
                      range(2 * len(target))]
        else:
            params = [{'name': f'Base {i + 1}', 'type': 'nominal', 'categories': RNA_BASES} for i in
                      range(len(target))]

        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(binary_mode=self.binary_mode, target=self.target)
