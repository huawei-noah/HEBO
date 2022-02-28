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

import matplotlib.pyplot as plt
import numpy as np
import os
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any, Type

from core.utils.build_in_seq.main import BUILD_IN_SEQ, RefObj
from core.action_space import Action, ACTION_SPACES
from core.design_groups import get_designs_path
from core.sessions.utils import get_design_prop
from utils.utils_plot import get_cummin, plot_mean_std
from utils.utils_save import get_storage_root, load_w_pickle, save_w_pickle


# DEPRECATED
class EDAExp(ABC):
    color = None
    linestyle = None

    def __init__(self, design_file: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 lut_inputs: int = 4,
                 ref_abc_seq: Optional[str] = None):
        """
        Args:
            design_file: path to the design
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """
        assert mapping in ('scl', 'fpga'), f"Mapping should be in [scl, fpga], got {mapping}"
        self.mapping = mapping
        assert os.path.exists(library_file), os.path.abspath(library_file)
        self.library_file = library_file
        self.abc_binary = abc_binary

        self.exec_time = 0
        self.design_file = design_file
        self.design_name = os.path.basename(design_file).split('.')[0]
        self.seq_length = seq_length

        self.action_space_id = action_space_id

        if ref_abc_seq is None:
            ref_abc_seq = 'init'  # evaluate initial design
        self.ref_abc_seq = ref_abc_seq
        biseq_cl = BUILD_IN_SEQ[ref_abc_seq]
        self.biseq = biseq_cl(library_file=self.library_file, design_file=self.design_file,
                              abc_binary=self.abc_binary)

        self.lut_inputs = lut_inputs
        ref_obj = RefObj(design_file=self.design_file, mapping=self.mapping, abc_binary=self.abc_binary,
                         library_file=self.library_file, lut_inputs=self.lut_inputs, ref_abc_seq=self.ref_abc_seq)

        self.ref_1, self.ref_2 = ref_obj.get_refs()

        self.action_space = self.get_action_space_()

    @abstractmethod
    def exists(self) -> bool:
        """ Check if experiment already exists """
        raise NotImplementedError()

    def get_action_space_(self) -> List[Action]:
        return self.get_action_space(action_space_id=self.action_space_id)

    @staticmethod
    def get_action_space(action_space_id: str) -> List[Action]:
        assert action_space_id in ACTION_SPACES, (action_space_id, list(ACTION_SPACES.keys()))
        return ACTION_SPACES[action_space_id]

    @abstractmethod
    def exp_id(self) -> str:
        raise NotImplementedError()

    def get_prop(self, seq: List[int], compute_init_stats: bool = False) -> Tuple[float, float, Dict[str, Any]]:
        sequence = [self.action_space[i].act_id for i in seq]
        return get_design_prop(seq=sequence, design_file=self.design_file, mapping=self.mapping,
                               library_file=self.library_file, abc_binary=self.abc_binary, lut_inputs=self.lut_inputs,
                               compute_init_stats=compute_init_stats)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return dict(
            mapping=self.mapping,
            design_file=self.design_file,
            design_name=self.design_name,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq
        )

    @property
    @abstractmethod
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        raise NotImplementedError()

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_name=self.design_name,
            ref_abc_seq=self.ref_abc_seq
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str, mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                         exp_id: str, design_name: str, ref_abc_seq: str = None) -> str:
        aux = f"{mapping}{f'-{lut_inputs}' if mapping == 'fpga' else ''}_{seq_length}_act-{action_space_id}"
        if ref_abc_seq != 'resyn2':
            aux += f"_{ref_abc_seq}"
        return os.path.join(get_storage_root(), meta_method_id, aux, exp_id, design_name)

    @property
    def obj1_id(self):
        if self.mapping == 'fpga':
            return f'lut_{self.lut_inputs}'
        elif self.mapping == 'scl':
            return 'area'
        else:
            raise ValueError(self.mapping)

    @property
    def obj2_id(self):
        if self.mapping == 'fpga':
            return f'levels'
        elif self.mapping == 'scl':
            return 'delay'
        else:
            raise ValueError(self.mapping)

    @property
    def action_space_length(self):
        return len(self.action_space)

    @staticmethod
    def plot_regret_qor(qors: np.ndarray, add_ref: bool = False, ax: Optional[Axes] = None,
                        exp_cls: Optional[Type['EDAExp']] = None,
                        **plot_kw) -> Axes:
        """
        Plot regret QoR curve

        Args:
            qors: array of qors obtained using some algorithm
            add_ref: whether to add initial QoR of 2 (QoR of the ref)
            ax: axis
            exp_cls: subclass of EDAExp used to get these results
            **plot_kw: plot kwargs

        Returns:
            ax: the axis
        """
        if ax is None:
            ax = plt.subplot()
        if 'c' not in plot_kw and 'color' not in plot_kw:
            plot_kw['c'] = exp_cls.color
        if 'linestyle' not in plot_kw:
            plot_kw['linestyle'] = exp_cls.linestyle
        qors = np.atleast_2d(qors)
        if add_ref:
            aux_qors = []
            for qor in qors:
                aux_qors = np.concatenate([np.array([2]), qor])
            qors = np.array(aux_qors)
        regret_qors = get_cummin(qors)
        ax = plot_mean_std(regret_qors, ax=ax, **plot_kw)
        return ax


class MultiEADExp:

    def __init__(self, designs_group_id: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 use_yosys: bool,
                 abc_binary: str, lut_inputs: int,
                 n_parallel: int = 1,
                 ref_abc_seq: Optional[str] = None):
        """
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            use_yosys: whether to use yosys-abc or abc_py
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """
        assert mapping in ('scl', 'fpga'), f"Mapping should be in [scl, fpga], got {mapping}"
        self.mapping = mapping
        assert os.path.exists(library_file), os.path.abspath(library_file)
        self.library_file = library_file
        self.abc_binary = abc_binary
        self.use_yosys = use_yosys

        self.exec_time = 0
        self.designs_group_id = designs_group_id
        self.design_files = get_designs_path(self.designs_group_id)
        self.design_names = list(
            map(lambda design_path: os.path.basename(design_path).split('.')[0], self.design_files))
        self.seq_length = seq_length

        self.action_space_id = action_space_id

        if ref_abc_seq is None:
            ref_abc_seq = 'init'  # evaluate initial design
        self.ref_abc_seq = ref_abc_seq

        biseq_cl = BUILD_IN_SEQ[ref_abc_seq]
        self.biseq = biseq_cl(library_file=self.library_file, design_file=self.design_files[0],
                              abc_binary=self.abc_binary)

        self.refs_1: List[float] = []
        self.refs_2: List[float] = []

        self.lut_inputs = lut_inputs

        refs = Parallel(n_jobs=n_parallel, backend="multiprocessing")(delayed(self.get_ref)(self.design_files[ind])
                                                                      for ind in tqdm(range(len(self.design_files))))

        for refs_1_2 in refs:
            self.refs_1.append(refs_1_2[0])
            self.refs_2.append(refs_1_2[1])

        self.action_space: List[Action] = self.get_action_space()

    @abstractmethod
    def exists(self) -> bool:
        """ Check if experiment already exists """
        raise NotImplementedError()

    def get_ref(self, design_file: str) -> Tuple[float, float]:
        """ Return either area and delay or lut and levels """

        ref_obj = RefObj(design_file=design_file, mapping=self.mapping, abc_binary=self.abc_binary,
                         library_file=self.library_file, lut_inputs=self.lut_inputs, ref_abc_seq=self.ref_abc_seq,
                         use_yosys=self.use_yosys)

        ref_1, ref_2 = ref_obj.get_refs()

        return ref_1, ref_2

    def get_action_space(self) -> List[Action]:
        assert self.action_space_id in ACTION_SPACES, (self.action_space_id, list(ACTION_SPACES.keys()))
        return ACTION_SPACES[self.action_space_id]

    @abstractmethod
    def exp_id(self) -> str:
        raise NotImplementedError()

    # def get_prop(self, seq: List[int]) -> Tuple[float, float]:
    #     sequence = [self.action_space[i].act_str for i in seq]
    #     return get_design_prop(seq=sequence, design_file=self.design_file, mapping=self.mapping,
    #                            library_file=self.library_file, abc_binary=self.abc_binary, lut_inputs=self.lut_inputs)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return dict(
            mapping=self.mapping,
            design_files_id=self.designs_group_id,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq,
            use_yosys=self.use_yosys
        )

    @property
    @abstractmethod
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        raise NotImplementedError()

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_abc_seq=self.ref_abc_seq
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str, mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                         exp_id: str, design_files_id: str, ref_abc_seq: str) -> str:
        return os.path.join(get_storage_root(), meta_method_id,
                            f"{mapping}{f'-{lut_inputs}' if mapping == 'fpga' else ''}"
                            f"_seq-{seq_length}_ref-{ref_abc_seq}"
                            f"_act-{action_space_id}",
                            exp_id,
                            design_files_id)

    @staticmethod
    def get_eval_ckpt_root_path(mapping: str, lut_inputs: int, use_yosys: bool, action_space_id: str) -> str:
        return os.path.join(get_storage_root(),
                            f"{mapping}{f'-{lut_inputs}' if mapping == 'fpga' else ''}{'_yosys' if use_yosys else ''}"
                            f"_{action_space_id}")

    @property
    def eval_ckpt_root_path(self) -> str:
        return self.get_eval_ckpt_root_path(mapping=self.mapping, lut_inputs=self.lut_inputs, use_yosys=self.use_yosys,
                                            action_space_id=self.action_space_id)

    @property
    def obj1_id(self):
        if self.mapping == 'fpga':
            return f'lut_{self.lut_inputs}'
        elif self.mapping == 'scl':
            return 'area'
        else:
            raise ValueError(self.mapping)

    @property
    def obj2_id(self):
        if self.mapping == 'fpga':
            return f'levels'
        elif self.mapping == 'scl':
            return 'delay'
        else:
            raise ValueError(self.mapping)

    @property
    def action_space_length(self):
        return len(self.action_space)


class MultiseqEADExp(MultiEADExp):

    def __init__(self, designs_group_id: str, seq_length: int, n_universal_seqs: int, mapping: str,
                 action_space_id: str, library_file: str, abc_binary: str, n_parallel: int = 1, lut_inputs: int = 4,
                 ref_abc_seq: Optional[str] = None):
        """
        Looking for `n_universal_seqs` universal sequences working for all circuits
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            n_universal_seqs: number of sequences
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            library_file: library file (asap7.lib)
            abc_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
        """
        super().__init__(
            designs_group_id=designs_group_id,
            seq_length=seq_length,
            mapping=mapping,
            action_space_id=action_space_id,
            library_file=library_file,
            abc_binary=abc_binary,
            n_parallel=n_parallel,
            lut_inputs=lut_inputs,
            ref_abc_seq=ref_abc_seq
        )
        self.n_universal_seqs = n_universal_seqs

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['n_universal_seqs'] = self.n_universal_seqs
        return config

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            mapping=self.mapping,
            lut_inputs=self.lut_inputs,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_abc_seq=self.ref_abc_seq,
            n_universal_seqs=self.n_universal_seqs
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str, mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                         exp_id: str, design_files_id: str, ref_abc_seq: str, n_universal_seqs: int) -> str:
        return os.path.join(get_storage_root(), meta_method_id,
                            f"{mapping}{f'-{lut_inputs}' if mapping == 'fpga' else ''}"
                            f"_seq-{seq_length}_ref-{ref_abc_seq}"
                            f"_act-{action_space_id}_n-univesal-{n_universal_seqs}",
                            exp_id,
                            design_files_id)


class Checkpoint:
    """
    Useful class for checkpointing (store the inputs tested so far and the ratios associated to first and second
        objectives for each input
     """

    def __init__(self, samples: np.ndarray, full_objs_1: np.ndarray, full_objs_2: np.ndarray):
        self.samples = samples
        self.full_objs_1 = full_objs_1
        self.full_objs_2 = full_objs_2
