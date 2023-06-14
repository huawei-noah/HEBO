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

from typing import Dict, List, Any, Optional

from mcbo.tasks.eda_seq_opt.utils.utils_operators import BOILS_MAPPING_OPERATORS, BOILS_PRE_MAPPING_OPERATORS, \
    BOILS_POST_MAPPING_OPERATORS

# ------------ Pre-mapping --------------- #

BOILS_PRE_MAPPING_ALGO_PARAMS = {
    'rewrite_wo_z': [{'name': 'rewrite_wo_z -l', 'type': 'bool'}],
    'rewrite_w_z': [{'name': 'rewrite_w_z -l', 'type': 'bool'}],
    'refactor_w_z': [
        {'name': 'refactor_w_z N', 'type': 'int', 'lb': 2, 'ub': 15},
        {'name': 'refactor_w_z -l', 'type': 'bool'},
    ],
    'refactor_wo_z': [
        {'name': 'refactor_wo_z N', 'type': 'int', 'lb': 2, 'ub': 15},
        {'name': 'refactor_wo_z -l', 'type': 'bool'},
    ],
    'resub': [{'name': 'resub K', 'type': 'int', 'lb': 5, 'ub': 8},
              {'name': 'resub N', 'type': 'int', 'lb': 0, 'ub': 3},
              {'name': 'resub -l', 'type': 'bool'},
              {'name': 'resub -z', 'type': 'bool'}],
    'balance': [{'name': 'balance -l', 'type': 'bool'},
                {'name': 'balance -d', 'type': 'bool'},
                {'name': 'balance -s', 'type': 'bool'},
                {'name': 'balance -x', 'type': 'bool'}],
    '&blut': [{'name': '&blut C', 'type': 'int', 'lb': 1, 'ub': 8},
              {'name': '&blut -a', 'type': 'bool'}],
    '&sopb': [{'name': '&sopb C', 'type': 'int', 'lb': 8, 'ub': 100}],
    '&dsdb': [{'name': '&dsdb C', 'type': 'int', 'lb': 8, 'ub': 100}],
    'fraig': [{'name': 'fraig -r', 'type': 'bool'}],
}

BOILS_PRE_MAPPING_ALGO_NO_PARAMS = {
    op.op_id: [] for op in BOILS_PRE_MAPPING_OPERATORS
}

# ------------ Mapping --------------- #

BOILS_MAPPING_ALGO_PARAMS = {
    'if': [
        {'name': 'if C', 'type': 'int', 'lb': 4, 'ub': 1024},
        {'name': 'if F', 'type': 'int', 'lb': 0, 'ub': 2},
        {'name': 'if A', 'type': 'int', 'lb': 0, 'ub': 4},
    ],
    'if -a': [
        {'name': 'if -a C', 'type': 'int', 'lb': 4, 'ub': 1024},
        {'name': 'if -a F', 'type': 'int', 'lb': 0, 'ub': 2},
        {'name': 'if -a A', 'type': 'int', 'lb': 0, 'ub': 4},
    ],
}

BOILS_MAPPING_ALGO_NO_PARAMS = {
    op.op_id: [] for op in BOILS_MAPPING_OPERATORS
}

# ------------ Post-mapping --------------- #

BOILS_POST_MAPPING_ALGO_PARAMS = {
    'speedup_if': [
        {'name': 'speedup_if speedup P', 'type': 'int', 'lb': 5, 'ub': 20},
        {'name': 'speedup_if speedup N', 'type': 'int', 'lb': 1, 'ub': 5},
        {'name': 'speedup_if if C', 'type': 'int', 'lb': 4, 'ub': 256},
        {'name': 'speedup_if if F', 'type': 'int', 'lb': 0, 'ub': 2},
    ],
    'mfs2_lutpack': [
        {'name': 'mfs2_lutpack mfs2 W', 'type': 'int', 'lb': 2, 'ub': 200},
        {'name': 'mfs2_lutpack mfs2 M', 'type': 'int', 'lb': 300, 'ub': 1000},
        {'name': 'mfs2_lutpack mfs2 D', 'type': 'int', 'lb': 0, 'ub': 20},
        {'name': 'mfs2_lutpack mfs2 -a', 'type': 'bool'},
        {'name': "mfs2_lutpack lutpack N", 'type': 'int', 'lb': 2, 'ub': 16},
        {'name': "mfs2_lutpack lutpack S", 'type': 'int', 'lb': 0, 'ub': 3},
        {'name': "mfs2_lutpack lutpack -z", 'type': 'bool'},
    ],
}

BOILS_POST_MAPPING_ALGO_NO_PARAMS = {
    op.op_id: [] for op in BOILS_POST_MAPPING_OPERATORS
}


# ------------------------------------------------------------------------------------ #


class OperatorHypSpace:

    def __init__(self,
                 pre_mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 post_mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 ):
        self.pre_mapping_operator_hyps = pre_mapping_operator_hyps
        self.mapping_operator_hyps = mapping_operator_hyps
        self.post_mapping_operator_hyps = post_mapping_operator_hyps
        self.all_hyps: Dict[str, List[Dict[str, Any]]] = {}
        self.all_hyps.update(self.pre_mapping_operator_hyps)
        self.all_hyps.update(self.mapping_operator_hyps)
        self.all_hyps.update(self.post_mapping_operator_hyps)


HYPERPARAMS_SPACES: Dict[str, OperatorHypSpace] = {
    'boils_hyp_op_space': OperatorHypSpace(
        pre_mapping_operator_hyps=BOILS_PRE_MAPPING_ALGO_PARAMS,
        mapping_operator_hyps=BOILS_MAPPING_ALGO_PARAMS,
        post_mapping_operator_hyps=BOILS_POST_MAPPING_ALGO_PARAMS
    ),
}


def get_operator_hyperparms_space(operator_hyperparams_space_id: Optional[str]) -> Optional[OperatorHypSpace]:
    if operator_hyperparams_space_id is None or operator_hyperparams_space_id == "":
        return None
    return HYPERPARAMS_SPACES[operator_hyperparams_space_id]
