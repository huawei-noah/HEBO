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

from joblib import Parallel, delayed

from core.algos.common_exp import MultiEADExp
from typing import Optional, List, Dict, Any
import os
import numpy as np
import time
import tqdm

from core.sessions.utils import get_design_prop
from utils.utils_misc import time_formatter, log
from utils.utils_save import save_w_pickle


class MultiTestRes:

    def __init__(self, objs_1: np.ndarray, objs_2: np.ndarray, qor_dict: Dict[str, float]):
        self.objs_1 = objs_1
        self.objs_2 = objs_2
        self.qor_dict = qor_dict


class MultiEADTestSeqExp(MultiEADExp):
    meta_method_id = 'multi-test'

    @property
    def method_id(self) -> str:
        return f'multi-test'

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 lut_inputs: int, use_yosys: bool,
                 seq_to_test: List[int], seq_origin: str,
                 n_parallel: int = 1,
                 ref_abc_seq: Optional[str] = None):
        """
        Args:
            designs_group_id: id of the designs group
            seq_to_test: sequence to evaluate by applying it to each design
            seq_origin: name of the method applied to get this sequence (just for the id)
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            use_yosys: whether to use yosys or abcRL package
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """

        super().__init__(designs_group_id=designs_group_id, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file,
                         abc_binary=abc_binary, n_parallel=n_parallel,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq, use_yosys=use_yosys)
        self.seq_origin = seq_origin
        self.seq_to_test = seq_to_test
        assert len(self.seq_to_test) == self.seq_length, (len(seq_to_test), self.seq_length)
        self.seq_ops_id_to_test: List[str] = [self.action_space[ind].act_id for ind in self.seq_to_test]

    def get_config(self) -> Dict[str, Any]:
        config = super(MultiEADTestSeqExp, self).get_config()
        config['seq_to_test'] = self.seq_to_test
        config['seq_ops_id_to_test'] = self.seq_ops_id_to_test
        config['use_yosys'] = self.use_yosys
        return config

    def exp_id(self):
        return self.get_exp_id(seq_to_test=self.seq_to_test, seq_origin=self.seq_origin, use_yosys=self.use_yosys)

    @staticmethod
    def get_exp_id(seq_to_test: List[int], seq_origin: str, use_yosys: bool):
        aux = f"test-{'-'.join(map(str, seq_to_test))}-{seq_origin}"
        if use_yosys:
            aux += f"_yosys"
        return aux

    def exp_path(self) -> str:
        return self.get_exp_path(
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_abc_seq=self.ref_abc_seq,
        )

    @staticmethod
    def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                     exp_id: str, design_files_id: str, ref_abc_seq: str):
        return MultiEADExp.get_exp_path_aux(
            meta_method_id=MultiEADTestSeqExp.meta_method_id,
            mapping=mapping,
            lut_inputs=lut_inputs,
            seq_length=seq_length,
            action_space_id=action_space_id,
            exp_id=exp_id,
            design_files_id=design_files_id,
            ref_abc_seq=ref_abc_seq
        )

    def get_obj(self, sequence: List[int], design_file: str, ref_1: float, ref_2: float):
        """ Return either area and delay or lut and levels """
        sequence = [(self.action_space[ind].act_id if not self.use_yosys else self.action_space[ind].act_str) for ind in
                    sequence]

        obj_1, obj_2, extra_info = get_design_prop(seq=sequence, design_file=design_file, mapping=self.mapping,
                                                   library_file=self.library_file, abc_binary=self.abc_binary,
                                                   lut_inputs=self.lut_inputs, compute_init_stats=False,
                                                   use_yosys=self.use_yosys)
        return obj_1 / ref_1, obj_2 / ref_2, extra_info

    def run(self, n_parallel: int = 1, verbose: bool = True) -> MultiTestRes:
        t = time.time()
        full_obj_1_s = np.zeros(len(self.design_files))  # obj_1 for each design file
        full_obj_2_s = np.zeros(len(self.design_files))  # obj_2 for each design file
        qor_dic = {}

        pbar = tqdm.tqdm(range(len(self.design_files)), desc=f"{self.designs_group_id} | {self.exp_id()}")
        objs = Parallel(n_jobs=n_parallel, backend="multiprocessing")(
            delayed(self.get_obj)(sequence=self.seq_to_test, design_file=self.design_files[k], ref_1=self.refs_1[k],
                                  ref_2=self.refs_2[k]) for k in pbar)

        for k, design_file in enumerate(self.design_files):
            full_obj_1_s[k], full_obj_2_s[k], _ = objs[k]
            qor_dic[os.path.basename(design_file).split('.')[0]] = objs[k][0] + objs[k][1]  # QoR

        QoR_mean = (full_obj_1_s + full_obj_2_s).mean()

        res = MultiTestRes(objs_1=full_obj_1_s, objs_2=full_obj_2_s, qor_dict=qor_dic)
        self.exec_time = time.time() - t
        if verbose:
            self.log(
                f"Took {time_formatter(self.exec_time)} to optimise {self.designs_group_id} "
                f"-> {self.designs_group_id} | Average QoR improvement -> {((2 - QoR_mean) / 2) * 100:.2f}%")
        return res

    def save_results(self, res: MultiTestRes) -> None:
        save_path = self.exp_path()
        self.log(f'{self.exp_id()} -> Save to {save_path}...')
        os.makedirs(save_path, exist_ok=True)

        # save execution time
        np.save(os.path.join(save_path, 'exec_time.npy'), np.array(self.exec_time))

        # save config
        save_w_pickle(self.get_config(), save_path, filename='config.pkl')

        # save res
        save_w_pickle(res, save_path, 'res.pkl')
        self.log(f'{self.exp_id()} -> Saved!')

    def exists(self) -> bool:
        """ Check if experiment already exists """
        save_path = self.exp_path()
        paths_to_check = [
            os.path.join(save_path, 'exec_time.npy'),
            os.path.join(save_path, 'config.pkl'),
            os.path.join(save_path, 'res.pkl')
        ]
        return np.all(list(map(lambda p: os.path.exists(p), paths_to_check)))

    def log(self, msg: str, end=None) -> None:
        log(msg, header=f"{self.method_id} - {self.seq_to_test} - {self.seq_origin} | {self.designs_group_id}", end=end)
