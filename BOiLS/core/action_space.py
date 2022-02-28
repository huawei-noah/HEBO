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

from subprocess import check_output
from typing import List, Dict


class Action:
    """ abc action """

    def __init__(self, act_id: str, act_str: str):
        """

        Args:
            act_id: action id
            act_str: string used to apply the action
        """
        self.act_id = act_id
        self.act_str = act_str

    def __repr__(self):
        return f"{self.act_id} -> {self.act_str}"


class ActionSimple(Action):
    """ Action for which act_str = act_id + ;  """

    def __init__(self, act_id: str):
        """

        Args:
            act_id: action id
        """
        super().__init__(act_id=act_id, act_str=act_id + ';')


class ActionCompo(Action):

    def __init__(self, act_id: str):
        """

        Args:
            act_id: action id
        """
        act_str = f'&get -n; {act_id}; &put;'
        super().__init__(act_id=act_id, act_str=act_str)


BALANCE = ActionSimple('balance')
REWRITE = ActionSimple('rewrite')
REWRITE_Z = ActionSimple('rewrite -z')
REFACTOR = ActionSimple('refactor')
REFACTOR_Z = ActionSimple('refactor -z')
RESUB = ActionSimple('resub')
RESUB_Z = ActionSimple('resub -z')
FRAIG = ActionSimple('fraig')
SOPB = ActionCompo('&sopb')
BLUT = ActionCompo('&blut')
DSDB = ActionCompo('&dsdb')
STRASH = ActionSimple('strash')

STD_ACTION_SPACE: List[Action] = [
    REWRITE,
    REWRITE_Z,
    REFACTOR,
    REFACTOR_Z,
    RESUB,
    RESUB_Z,
    BALANCE
]

EXTENDED_ACTION_SPACE: List[Action] = [
    REWRITE,
    REWRITE_Z,
    REFACTOR,
    REFACTOR_Z,
    RESUB,
    RESUB_Z,
    BALANCE,
    FRAIG,
    SOPB,
    BLUT,
    DSDB
]

STRASH_EXTENDED_ACTION_SPACE: List[Action] = [
    REWRITE,
    REWRITE_Z,
    REFACTOR,
    REFACTOR_Z,
    RESUB,
    RESUB_Z,
    BALANCE,
    FRAIG,
    SOPB,
    BLUT,
    DSDB,
    STRASH
]

ACTION_SPACES: Dict[str, List[Action]] = {
    'standard': STD_ACTION_SPACE,
    'extended': EXTENDED_ACTION_SPACE,
    'strash_extended': STRASH_EXTENDED_ACTION_SPACE

}

if __name__ == '__main__':
    import sys
    from pathlib import Path
    import os

    ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
    sys.path[0] = ROOT_PROJECT
    from DRiLLS.drills.fpga_session import FPGASession

    for action_space in (EXTENDED_ACTION_SPACE, STD_ACTION_SPACE):
        design_file = os.path.join(ROOT_PROJECT, 'data/epfl-benchmark/arithmetic/adder.v')
        abc_binary = 'yosys-abc'
        library_file = os.path.join(ROOT_PROJECT, './asap7.lib')

        abc_command = f'read {library_file}; '
        abc_command += f'read {design_file}; '
        abc_command += 'strash; '
        for action in action_space:
            abc_command += action.act_str
        abc_command += 'if -K 4;'
        abc_command += 'print_stats;'
        print(abc_command)
        proc = check_output([abc_binary, '-c', abc_command])
        print(FPGASession._get_metrics(proc))
