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

import abc
import os
from typing import List, Tuple, Dict, Union, Type, Any

from core.sessions.utils import get_design_prop
from utils.utils_save import get_storage_root, load_w_pickle, save_w_pickle


class BuildInSeq(abc.ABC):
    sequence: List[str]

    def __init__(self, library_file: str, design_file: str, abc_binary: str):
        self.library_file = library_file
        self.design_file = design_file
        self.abc_binary = abc_binary

    def fpga(self, lut_inputs: int, compute_init_stats: bool, use_yosys: bool, verbose: bool = False) \
            -> Tuple[int, int, Dict[str, Any]]:
        """ Return lut-6 and levels after application of predifined sequence """
        return get_design_prop(
            library_file=self.library_file,
            mapping='fpga',
            design_file=self.design_file,
            abc_binary=self.abc_binary,
            lut_inputs=lut_inputs,
            seq=self.sequence,
            compute_init_stats=compute_init_stats,
            use_yosys=use_yosys,
            verbose=verbose,
        )

    @staticmethod
    def seq_length() -> int:
        raise NotImplementedError()


class Resyn(BuildInSeq):
    sequence = [
        'balance',
        'rewrite',
        'rewrite -z',
        'balance',
        'rewrite -z',
        'balance'
    ]

    def __init__(self, library_file: str, design_file: str, abc_binary: str):
        """
            balance; rewrite; rewrite -z; balance; rewrite -z; balance
        """

        super().__init__(library_file, design_file, abc_binary)

    @staticmethod
    def seq_length() -> int:
        return len(Resyn.sequence)


resyn2_seq = [
    'balance',
    'rewrite',
    'refactor',
    'balance',
    'rewrite',
    'rewrite -z',
    'balance',
    'refactor -z',
    'rewrite -z',
    'balance'
]


class Resyn2(BuildInSeq):
    sequence = resyn2_seq

    def __init__(self, library_file: str, design_file: str, abc_binary: str):
        """
            balance; rewrite; refactor; balance; rewrite; rewrite –z; balance; refactor –z; rewrite –z; balance;
        """

        super().__init__(library_file, design_file, abc_binary)

    @staticmethod
    def seq_length() -> int:
        return len(Resyn2.sequence)


class InitDesign(BuildInSeq):
    sequence = []

    def __init__(self, library_file: str, design_file: str, abc_binary: str):
        """
            No action, evaluate initial design
        """

        super().__init__(library_file, design_file, abc_binary)

    @staticmethod
    def seq_length() -> int:
        return len(InitDesign.sequence)


BUILD_IN_SEQ: Dict[str, Union[Type[InitDesign], Type[Resyn], Type[Resyn2]]] = dict(
    init=InitDesign,
    resyn=Resyn,
    resyn2=Resyn2
)


class RefObj:

    def __init__(self, design_file: str, mapping: str, abc_binary: str, library_file: str,
                 lut_inputs: int, use_yosys: bool,
                 ref_abc_seq: str):
        """
            Args:
                design_file: path to the design
                mapping: either scl of fpga mapping
                library_file: library file (asap7.lib)
                abc_binary: (probably yosys-abc)
                use_yosys: whether to use yosys-abc or abc_py
                lut_inputs: number of LUT inputs (2 < num < 33)
                ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            """
        self.design_file = design_file
        self.mapping = mapping
        self.abc_binary = abc_binary
        self.library_file = library_file
        self.lut_inputs = lut_inputs
        self.ref_abc_seq = ref_abc_seq
        self.use_yosys = use_yosys

        self.design_name = os.path.basename(design_file).split('.')[0]

    def get_config(self) -> Dict[str, Any]:
        return dict(
            mapping=self.mapping,
            design_file=self.design_file,
            design_name=self.design_name,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq
        )

    def ref_path(self) -> str:
        path_id = f"{self.mapping}{f'-{self.lut_inputs}' if self.mapping == 'fpga' else ''}"
        if not self.use_yosys:
            path_id += '-abcpy'
        return os.path.join(get_storage_root(), 'refs', self.ref_abc_seq, path_id, self.design_name)

    def get_refs(self) -> Tuple[float, float]:
        if os.path.exists(os.path.join(self.ref_path(), 'refs.pkl')):
            refs = load_w_pickle(self.ref_path(), 'refs.pkl')
            return refs['ref_1'], refs['ref_2']

        biseq_cl = BUILD_IN_SEQ[self.ref_abc_seq]
        biseq = biseq_cl(library_file=self.library_file, design_file=self.design_file,
                         abc_binary=self.abc_binary)

        if self.mapping == 'scl':
            raise ValueError(self.mapping)
            # ref_1, ref_2 = biseq.scl()
        elif self.mapping == 'fpga':
            ref_1, ref_2, extra_info = biseq.fpga(lut_inputs=self.lut_inputs, verbose=True, compute_init_stats=False,
                                                  use_yosys=self.use_yosys)
        else:
            raise ValueError(self.mapping)
        os.makedirs(self.ref_path(), exist_ok=True)
        ref_obj = dict(ref_1=ref_1, ref_2=ref_2, config=self.get_config(), exec_time=extra_info['exec_time'])
        save_w_pickle(ref_obj, self.ref_path(), 'refs.pkl')
        return ref_1, ref_2


if __name__ == '__main__':
    print('; '.join(resyn2_seq))
