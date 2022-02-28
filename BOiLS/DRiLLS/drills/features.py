#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

# 2021.11.10-Minor reformatting
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import re
from collections import defaultdict
from multiprocessing import Process, Manager
from subprocess import check_output

import numpy as np


def yosys_stats(design_file, yosys_binary, stats):
    yosys_command = "read_blif " + design_file + "; stat"
    try:
        proc = check_output([yosys_binary, '-QT', '-p', yosys_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'Number of wires' in line:
                stats['number_of_wires'] = int(line.strip().split()[-1])
            if 'Number of public wires' in line:
                stats['number_of_public_wires'] = int(line.strip().split()[-1])
            if 'Number of cells' in line:
                stats['number_of_cells'] = float(line.strip().split()[-1])
            if '$and' in line:
                stats['ands'] = float(line.strip().split()[-1])
            if '$or' in line:
                stats['ors'] = float(line.strip().split()[-1])
            if '$not' in line:
                stats['nots'] = float(line.strip().split()[-1])

        # catch some design special cases
        if 'ands' not in stats:
            stats['ands'] = 0.0
        if 'ors' not in stats:
            stats['ors'] = 0.0
        if 'nots' not in stats:
            stats['nots'] = 0.0
    except Exception as e:
        print(e)
        return None
    return stats


def abc_stats(design_file, abc_binary, stats):
    abc_command = "read " + design_file + "; print_stats"
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
                stats['input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
                stats['output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())

                ob = re.search(r'edge *= *[0-9]+', line)
                stats['edges'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'nd *= *[0-9]+', line)
                stats['nodes'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lev *= *[0-9]+', line)
                stats['levels'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lat *= *[0-9]+', line)
                stats['latches'] = int(ob.group().split('=')[1].strip())
    except Exception as e:
        print(e)
        return None

    return stats


def extract_features(design_file, yosys_binary='yosys', abc_binary='abc'):
    """
    Returns features of a given circuit as a tuple.
    Features are listed below
    """
    manager = Manager()
    stats = manager.dict()
    p1 = Process(target=yosys_stats, args=(design_file, yosys_binary, stats))
    p2 = Process(target=abc_stats, args=(design_file, abc_binary, stats))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # normalized features
    features = defaultdict(float)

    # (1) - number of input/output pins
    features['input_pins'] = stats['input_pins']
    features['output_pins'] = stats['output_pins']

    # (2) - number of nodes and edges
    features['number_of_nodes'] = stats['number_of_cells']
    features['number_of_edges'] = stats['edges']

    # (3) - number of levels
    features['number_of_levels'] = stats['levels']

    # (4) - number of latches
    features['number_of_latches'] = stats['latches']

    # (5) - gate types percentages
    features['percentage_of_ands'] = stats['ands'] / stats['number_of_cells']
    # features['percentage_of_ors'] = stats['ors'] / stats['number_of_cells']
    features['percentage_of_nots'] = stats['nots'] / stats['number_of_cells']

    return np.array([
        features['input_pins'] / features['output_pins'],
        features['number_of_nodes'],
        features['number_of_edges'],
        features['number_of_levels'],
        features['number_of_latches'],
        features['percentage_of_ands'],
        # features['percentage_of_ors'],
        features['percentage_of_nots']
    ])
