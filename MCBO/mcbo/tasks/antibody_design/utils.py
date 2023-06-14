# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import pathlib
import re
import subprocess
import zipfile
from io import BytesIO
from itertools import groupby
from typing import Optional, Dict

import numpy as np
import pandas as pd
import requests

MAX_AA_COUNT = 5  # maximum number of consecutive AAs
N_glycosylation_pattern = 'N[^P][ST][^P]'


def get_valid_antigens(AbsolutNoLib_dir: str):
    output = subprocess.run([os.path.join(AbsolutNoLib_dir, 'AbsolutNoLib'), 'listAntigens'], capture_output=True,
                            text=True)
    antigens = []
    for antigen_str in output.stdout.split('\n')[:-2]:
        antigens.append(antigen_str.split('\t')[1])

    return antigens


def get_AbsolutNoLib_dir(path_to_AbsolutNoLib: Optional[str] = None) -> str:
    if path_to_AbsolutNoLib is None:
        path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'path_to_AbsolutNoLib.txt')
        try:
            f = open(path, "r")
            AbsolutNoLib_dir = f.readlines()[0]
            if AbsolutNoLib_dir[-1] == '\n':
                AbsolutNoLib_dir = AbsolutNoLib_dir[:-1]
            f.close()

        except FileNotFoundError as _:
            abs_path_to_absolut_no_lib = '/my/absolute/path/to/AbsolutNoLib'
            error_message = f'\n\n------ Friendly first run error message ------\n\n' \
                            f'File {path} not found. \n\n' \
                            f'   --> Please create it and fill it with one line describing the absolute path to the ' \
                            f'AbsulutNoLib executable e.g. by running\n' \
                            f"\techo '{abs_path_to_absolut_no_lib}' > {path}\n" \
                            f'\n and then rerun your program.'
            raise FileNotFoundError(error_message)
    else:
        AbsolutNoLib_dir = path_to_AbsolutNoLib
    if AbsolutNoLib_dir.split('/')[-1] in ['AbsolutNoLib', 'Absolut']:
        AbsolutNoLib_dir = os.path.join('/', *AbsolutNoLib_dir.split('/')[:-1])

    assert os.path.exists(
        os.path.join(AbsolutNoLib_dir,
                     'AbsolutNoLib')) or os.path.exists(
        os.path.join(AbsolutNoLib_dir,
                     'Absolut')), f'AbsolutNoLib can\'t be found in provided directory {AbsolutNoLib_dir},' \
                                  f' check path specified in {path_to_AbsolutNoLib}'
    return AbsolutNoLib_dir


def download_precomputed_antigen_structure(AbsolutNoLib_dir: str, antigen: str, num_cpus: Optional[int] = 1,
                                           first_cpu: Optional[int] = 0):
    print('Checking if antigen precomputed structures are downloaded ... ')

    os.makedirs(os.path.join(AbsolutNoLib_dir, 'antigen_data', f'{antigen}'), exist_ok=True)

    assert (num_cpus is not None and first_cpu is not None) or (
            isinstance(num_cpus, int) and isinstance(first_cpu, int))

    if num_cpus > 0:
        absolut_run_command = ['taskset', '-c', f"{first_cpu}-{first_cpu + num_cpus}", "./../../AbsolutNoLib",
                               'repertoire', antigen, f"TempCDR3_{antigen}.txt", str(num_cpus)]
    else:
        if os.path.exists("./../../AbsolutNoLib"):
            ex = "./../../AbsolutNoLib"
        elif os.path.exists("./../../Absolut"):
            ex = "./../../Absolut"
        else:
            raise ValueError()
        absolut_run_command = [ex, 'repertoire', antigen, f"TempCDR3_{antigen}.txt"]

    current_dir = os.getcwd()
    os.chdir(os.path.join(AbsolutNoLib_dir, 'antigen_data', f'{antigen}'))

    seq = 'CARAAHKLARIPK'  # Random sequence

    sequences = []

    try:
        with open(f'TempCDR3_{antigen}.txt', 'w') as f:
            line = f"{1}\t{seq}\n"
            f.write(line)
            sequences.append(seq)
    except PermissionError:
        print(os.getcwd())
        raise

    repertoire_output = subprocess.run(absolut_run_command, capture_output=True, text=True)

    download_err_message = 'ERR: the list of binding structures for this antigen could not been found in this folder...'
    if repertoire_output.stdout.split('\n')[4] == download_err_message:
        print(repertoire_output.stdout.split('\n'))
        if os.path.exists("./../../AbsolutNoLib"):
            ex = "./../../AbsolutNoLib"
        elif os.path.exists("./../../Absolut"):
            ex = "./../../Absolut"
        else:
            raise ValueError()
        download_command = subprocess.run([ex, 'info_fileNames', f'{antigen}'], capture_output=True,
                                          text=True)
        download_command = download_command.stdout.split('\n')[3]
        assert download_command.split(' ')[0] == 'wget'

        # Fix corrupted download links
        corrupted_download_link = download_command.split(' ')[1]
        filename = corrupted_download_link.split('http://philippe-robert.com/Absolut/Structures/')[1]
        fixed_download_link = 'https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/NS9603K/pprobert/AbsolutOnline/Structures/' + \
                              filename

        # Download the zip file
        print(f'Downloading precomputed {antigen} structure ... {fixed_download_link}.zip in {os.getcwd()}')

        r = requests.get(fixed_download_link + '.zip', stream=True)
        assert r.ok, 'Download unsuccessful...'
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall()

        print(f'Requesting all possible receptor structures for antigen {antigen} ...')
        repertoire_output = subprocess.run(absolut_run_command, capture_output=True, text=True)

    if num_cpus is not None:
        # Remove all created files and change the working directory to what it was
        for i in range(num_cpus):
            os.remove(f"TempBindingsFor{antigen}_t{i}_Part1_of_1.txt")
    else:
        os.remove(f'TempBindingsFor{antigen}_t0_Part1_of_1.txt')

    os.remove(f'{antigen}FinalBindings_Process_1_Of_1.txt')
    os.remove(f'TempCDR3_{antigen}.txt')
    os.chdir(current_dir)


def compute_developability_scores(x: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    charge = np.zeros((len(x), 1))
    n_gly_seq = np.zeros((len(x), 1)).astype(bool)
    max_count = np.zeros((len(x), 1)).astype(int)
    for i in range(len(x)):
        seq = ''.join(j for j in x.iloc[i].array)
        # Compute the charge
        for aa in seq:
            charge[i, 0] += int(aa == 'R' or aa == 'K') + 0.1 * int(aa == 'H') - int(aa == 'D' or aa == 'E')
        # Check for the presence of the N-X-S/T pattern. This looks for the single letter code N, followed by any
        # character that is not P, followed by either an S or a T, followed by any character that is not a P. Source
        # https://towardsdatascience.com/using-regular-expression-in-genetics-with-python-175e2b9395c2
        if re.search(N_glycosylation_pattern, seq):
            n_gly_seq[i, 0] = True
        else:
            n_gly_seq[i, 0] = False
        # Maximum number of the same subsequent AAs
        max_count[i, 0] = max([sum(1 for _ in group) for _, group in groupby(seq)])
    return charge, n_gly_seq, max_count


def get_charge(x: Dict[str, str]) -> float:
    seq = ''.join(x[f'Amino acid {i + 1}'] for i in range(len(x)))
    charge = 0
    for aa in seq:
        charge += int(aa == 'R' or aa == 'K') + 0.1 * int(aa == 'H') - int(aa == 'D' or aa == 'E')

    return charge


def check_pattern(x: Dict[str, str]) -> float:
    seq = ''.join(x[f'Amino acid {i + 1}'] for i in range(len(x)))
    # Check for the presence of the N-X-S/T pattern. This looks for the single letter code N, followed by any
    # character that is not P, followed by either an S or a T, followed by any character that is not a P. Source
    # https://towardsdatascience.com/using-regular-expression-in-genetics-with-python-175e2b9395c2
    if re.search(N_glycosylation_pattern, seq):
        n_gly_seq = 1
    else:
        n_gly_seq = 0

    return n_gly_seq


def get_max_count(x: Dict[str, str]) -> float:
    seq = ''.join(x[f'Amino acid {i + 1}'] for i in range(len(x)))
    # Maximum number of the same subsequent AAs
    max_count = max([sum(1 for _ in group) for _, group in groupby(seq)])

    return max_count


def check_constraint_satisfaction(x: pd.DataFrame) -> np.ndarray:
    charge, n_gly_seq, max_count = compute_developability_scores(x)

    charge_constraint = np.logical_and(-2 <= charge, charge <= 2)
    no_n_gly_seq_constraint = np.logical_not(n_gly_seq)
    max_count_constraint = max_count <= MAX_AA_COUNT

    return np.logical_and(charge_constraint, np.logical_and(no_n_gly_seq_constraint, max_count_constraint))
