import os
import subprocess
from typing import List

import numpy as np
import pandas as pd


def evaluate_seq_absolut(sequences: List, pdb_id: str, AbsolutNoLib_dir: str, save_dir: str,
                         first_cpu: int = 0, num_cpus: int = 70) -> np.ndarray:
    # Change working directory
    current_dir = os.getcwd()
    if not os.path.exists(os.path.join(save_dir, 'antigen_data', pdb_id)):
        os.makedirs(os.path.join(save_dir, 'antigen_data', pdb_id))
    os.chdir(os.path.join(save_dir, 'antigen_data', pdb_id))
    pid = os.getpid()

    with open(f'TempCDR3_{pdb_id}_pid_{pid}.txt', 'w') as f:
        for i, seq in enumerate(sequences):
            line = f"{i + 1}\t{seq}\n"
            f.write(line)

    _ = subprocess.run(
        ['taskset', '-c', f"{first_cpu}-{first_cpu + num_cpus}",
         os.path.join(AbsolutNoLib_dir, 'AbsolutNoLib'), 'repertoire', pdb_id, f"TempCDR3_{pdb_id}_pid_{pid}.txt",
         str(num_cpus)], capture_output=True, text=True)

    data = pd.read_csv(f"{pdb_id}FinalBindings_Process_1_Of_1.txt", sep='\t', skiprows=1)

    # Add an extra column to ensure that ordering will be ok after groupby operation
    data['sequence_idx'] = data.apply(lambda row: int(row.ID_slide_Variant.split("_")[0]), axis=1)
    energy = data.groupby(by=['sequence_idx']).min(['Energy'])
    min_energy = energy['Energy'].values.reshape(-1, 1)

    # Remove all created files and change the working directory to what it was
    for i in range(num_cpus):
        os.remove(f"TempBindingsFor{pdb_id}_t{i}_Part1_of_1.txt")
    os.remove(f"TempCDR3_{pdb_id}_pid_{pid}.txt")

    os.remove(f"{pdb_id}FinalBindings_Process_1_Of_1.txt")
    os.chdir(current_dir)
    return min_energy