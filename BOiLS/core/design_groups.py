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
from typing import List, Dict

import numpy as np

from utils.utils_save import get_storage_data_root

DATA_PATH = os.path.join(get_storage_data_root(), 'benchmark_blif')

EPFL_ARITHMETIC = ['hyp', 'div', 'log2', 'multiplier', 'sqrt', 'square', 'sin', 'bar', 'adder', 'max']

DESIGN_GROUPS: Dict[str, List[str]] = {
    'epfl_arithmetic': EPFL_ARITHMETIC,
}

for design in EPFL_ARITHMETIC:
    DESIGN_GROUPS[design] = [design]

AUX_TEST_GP = ['adder', 'bar']
AUX_TEST_ABC_GRAPH = ['adder', 'sin']

DESIGN_GROUPS['aux_test_designs_group'] = AUX_TEST_GP
DESIGN_GROUPS['aux_test_abc_graph'] = AUX_TEST_ABC_GRAPH


def get_designs_path(designs_id: str, frac_part: str = None) -> List[str]:
    """ Get list of filepaths to designs """

    designs_filepath: List[str] = []
    for design_id in DESIGN_GROUPS[designs_id]:
        designs_filepath.append(os.path.join(DATA_PATH, f'{design_id}.blif'))
    if frac_part is None:
        s = slice(0, len(designs_filepath))
    else:
        i, j = map(int, frac_part.split('/'))
        assert j > 0 and i > 0, (i, j)
        step = int(np.ceil(len(designs_filepath) / j))
        s = slice((i - 1) * step, i * step)

    return designs_filepath[s]


if __name__ == '__main__':

    designs_id_ = 'test_designs_group'
    N = 6
    for n in range(1, N + 1):
        frac = f'{n}/{N}'
        print(f'{frac} -----> ', end='')
        print(get_designs_path(designs_id=designs_id_, frac_part=frac))
