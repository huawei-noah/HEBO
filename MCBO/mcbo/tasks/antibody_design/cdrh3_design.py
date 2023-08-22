# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import subprocess
from typing import Optional, List, Dict, Callable, Any

import numpy as np
import pandas as pd

from mcbo.tasks import TaskBase
from mcbo.tasks.antibody_design.utils import get_AbsolutNoLib_dir, download_precomputed_antigen_structure, \
    get_valid_antigens, compute_developability_scores, check_pattern, get_max_count, MAX_AA_COUNT, get_charge


class CDRH3Design(TaskBase):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                   'W', 'Y']
    amino_acid_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    idx_to_amino_acid = {value: key for key, value in amino_acid_to_idx.items()}

    @property
    def name(self) -> str:
        return f'{self.antigen} Antibody Design'

    def __init__(self, antigen: str = '1ADQ_A', cdrh3_length: int = 11, num_cpus: int = 10, first_cpu: int = 0,
                 absolut_dir: Optional[str] = None):
        super(CDRH3Design, self).__init__()
        self.num_cpus = num_cpus
        self.first_cpu = first_cpu
        self.antigen = antigen
        self.cdrh3_length = cdrh3_length

        self.AbsolutNoLib_dir = get_AbsolutNoLib_dir(absolut_dir)
        self.valid_antigens = get_valid_antigens(self.AbsolutNoLib_dir)
        self.need_to_check_precomputed_antigen_structure = True
        assert antigen in self.valid_antigens, f'Specified antigen is not valid. Please choose of from: \n\n {self.valid_antigens}'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:

        if self.need_to_check_precomputed_antigen_structure:
            download_precomputed_antigen_structure(self.AbsolutNoLib_dir, self.antigen)
            self.need_to_check_precomputed_antigen_structure = False

        assert os.path.exists(os.path.join(self.AbsolutNoLib_dir, 'antigen_data', f'{self.antigen}'))

        # Change working directory
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.AbsolutNoLib_dir, 'antigen_data', f'{self.antigen}'))
        pid = os.getpid()

        sequences = []
        with open(f'TempCDR3_{self.antigen}_pid_{pid}.txt', 'w') as f:
            for i in range(len(x)):
                seq = x.iloc[i]
                seq = ''.join(aa for aa in seq)
                line = f"{i + 1}\t{seq}\n"
                f.write(line)
                sequences.append(seq)

        _ = subprocess.run(
            ['taskset', '-c', f"{self.first_cpu}-{self.first_cpu + self.num_cpus}",
             "./../../AbsolutNoLib", 'repertoire', self.antigen, f"TempCDR3_{self.antigen}_pid_{pid}.txt",
             str(self.num_cpus)], capture_output=True, text=True)

        data = pd.read_csv(f"{self.antigen}FinalBindings_Process_1_Of_1.txt", sep='\t', skiprows=1)

        # Add an extra column to ensure that ordering will be ok after groupby operation
        data['sequence_idx'] = data.apply(lambda row: int(row.ID_slide_Variant.split("_")[0]), axis=1)
        energy = data.groupby(by=['sequence_idx']).min(['Energy'])
        min_energy = energy['Energy'].values.reshape(-1, 1)

        # Remove all created files and change the working directory to what it was
        for i in range(self.num_cpus):
            os.remove(f"TempBindingsFor{self.antigen}_t{i}_Part1_of_1.txt")
        os.remove(f"TempCDR3_{self.antigen}_pid_{pid}.txt")

        os.remove(f"{self.antigen}FinalBindings_Process_1_Of_1.txt")
        os.chdir(current_dir)
        return min_energy

    @staticmethod
    def compute_developability_scores(x: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
        charge, n_gly_seq, max_count = compute_developability_scores(x)
        return charge, n_gly_seq, max_count

    @staticmethod
    def get_static_search_space_params(cdrh3_length: int) -> List[Dict[str, Any]]:

        params = [{'name': f'Amino acid {i + 1}', 'type': 'nominal', 'categories': CDRH3Design.amino_acids} for i in
                  range(cdrh3_length)]

        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(cdrh3_length=self.cdrh3_length)

    @property
    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return self.get_input_constraints()

    @staticmethod
    def get_input_constraints() -> List[Callable[[Dict], bool]]:
        def charge_constr(x: Dict[str, str]) -> bool:
            charge = get_charge(x)
            return -2 <= charge <= 2

        def pattern_constr(x: Dict[str, str]) -> bool:
            return not check_pattern(x)

        def max_count_constr(x: Dict[str, str]) -> bool:
            return get_max_count(x) <= MAX_AA_COUNT

        return [charge_constr, pattern_constr, max_count_constr]
