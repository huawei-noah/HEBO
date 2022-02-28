# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary 
# forms, with or without modification, are permitted provided that the following conditions are met: 
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer. 
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided with the distribution. 
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission. 
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

import os
import re
from collections import defaultdict
from multiprocessing import Process, Manager

from subprocess import check_output
from threading import Thread

from typing import Optional, Tuple, Dict

import numpy as np


class Res:
    """ Auxiliary class to mimic pymoo format """

    def __init__(self, X: np.ndarray, F: np.ndarray, history_x: Optional[np.ndarray] = None,
                 history_f: Optional[np.ndarray] = None):
        """

        Args:
            X: best points (pareto front if multi-objective)
            F: function values (shape: (n_points, n_obj_functions)
            history_x: all
        """
        self.X = X
        self.F = F
        self.history_x = history_x
        self.history_f = history_f


def get_history_values_from_res(res: Res) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return an array

    Args:
        res: pymoo result object having `history` field

    Returns:
        X: array of inputs (-1, action_space_size)
        Y: array of obj values (-1, 2)
    """
    X = res.history_x
    Y = res.history_f
    assert Y.ndim == 2, Y.ndim
    assert X.shape == (Y.shape[0], X.shape[-1]), (Y.shape[0], X.shape[-1])
    return X, Y


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.logical_or(np.any(costs[is_efficient] < c, axis=1),
                                                       np.all(costs[is_efficient] == costs[i],
                                                              axis=1))  # Keep any point with a lower cost
    return is_efficient


def pareto_score(pareto_front: np.ndarray):
    """ Compute the score associated to a pareto front (for a 2-objective minimisation task)
    Args:
        pareto_front: np.ndarray of shape (n, 2) containing the 2 objectives at the n points on the pareto front

    Returns:
         score: area under pareto front
    """
    assert np.all(pareto_front > 0)
    assert pareto_front.ndim == 2 and pareto_front.shape[1] == 2, pareto_front.shape

    inds = pareto_front[:, 0].argsort()
    aux = pareto_front[inds]
    i = 0
    x = np.array([0, *aux[:, 0]])
    y = np.array([aux[0, 1], *aux[:, 1]])
    return np.trapz(y, x=x, axis=0)

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


def extract_features(design_file, yosys_binary='yosys', abc_binary='yosys-abc') -> Dict[str, float]:
    """
    Returns features of a given circuit as a tuple.
    Features are listed below
    """

    try:
        manager = Manager()
        stats = manager.dict()
        p2 = Process(target=abc_stats, args=(design_file, abc_binary, stats))
        p2.start()
        p2.join()
    except AssertionError:


        stats = {}
        thread = Thread(target=abc_stats, args=(design_file, abc_binary, stats))

        # thread.daemon = True
        thread.start()
        thread.join()

    # normalized features
    features = defaultdict(float)

    # (1) - number of input/output pins
    features['input_pins'] = stats['input_pins']
    features['output_pins'] = stats['output_pins']

    # (2) - number of nodes and edges
    features['number_of_edges'] = stats['edges']

    # (3) - number of levels
    features['number_of_levels'] = stats['levels']

    # (4) - number of latches
    features['number_of_latches'] = stats['latches']

    # (5) - gate types percentages
    # features['percentage_of_nots'] = stats['nots'] / stats['number_of_cells']

    # (6) - num nodes
    features['nodes'] = stats['nodes']

    return features


def get_design_name(design_filepath: str) -> str:
    return os.path.basename(design_filepath).split('.')[0]


class StateDesign:
    """ Data class whose fields are main characteristics of designs"""

    def __init__(self, n_inputs: float, n_outputs: float, and_nodes: float, levels: float, edges: float,
                 obj_1: float, obj_2: float):
        self.n_outputs = n_outputs
        self.and_nodes = and_nodes
        self.levels = levels
        self.edges = edges
        self.obj_1 = obj_1
        self.obj_2 = obj_2
        self.n_inputs = n_inputs

    def __repr__(self):
        s = 'State:\n'
        s += f'\t- n_out: {self.n_outputs}'
        s += f'\t- n_in: {self.n_inputs}'
        s += f'\t- n_nodes: {self.and_nodes}'
        s += f'\t- levels: {self.levels}'
        s += f'\t- edges: {self.edges}'
        s += f'\t- obj_1: {self.obj_1}'
        s += f'\t- obj_2: {self.obj_2}'
        return s
