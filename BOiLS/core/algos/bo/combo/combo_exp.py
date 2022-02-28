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
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from resources.COMBO.experiments.exp_utils import sample_init_points
from utils.utils_misc import log
from core.algos.common_exp import EDAExp
from core.algos.utils import is_pareto_efficient, Res, get_history_values_from_res
from core.sessions.utils import get_design_prop
from utils.utils_save import save_w_pickle


class COMBORes(Res):
    pass


class COMBOExp(EDAExp):
    """ Class associated to COMBO to solve QoR minimization: https://arxiv.org/pdf/1902.00448.pdf """

    color = 'cyan'

    def __init__(self, design_file: str, seq_length: int, mapping: str, action_space_id: str,
                 library_file: str,
                 abc_binary: str,
                 seed: int, n_initial: int, lamda: Optional[float],
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
            seed: reproducibility seed
            n_initial: number of initial points to test before building first surrogate model
            lamda: COMBO parameter
        """

        super().__init__(design_file=design_file, seq_length=seq_length, mapping=mapping,
                         action_space_id=action_space_id, library_file=library_file, abc_binary=abc_binary,
                         lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq)
        self.seed = seed
        np.random.seed(self.seed)
        self.n_initial = n_initial
        self.lamda = lamda

        self.ref_time = time.time()

        self.samples_X = []
        self.obj_1_s = []
        self.obj_2_s = []

        self.n_vertices = np.array([self.action_space_length] * self.seq_length)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init,
                                         sample_init_points(self.n_vertices,
                                                            self.n_initial - self.suggested_init.size(0),
                                                            self.seed).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = str(seed).zfill(4)
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.ones((n_v, n_v)) - torch.diag(torch.ones(n_v))
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        self.n_evals = 0

    def get_config(self) -> Dict[str, Any]:
        config = super(COMBOExp, self).get_config()
        config['seed'] = self.seed
        config['n_initial'] = self.n_initial
        config['lamda'] = self.lamda
        return config

    @property
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        return 'BO'

    @property
    def method_id(self) -> str:
        """ Id for the method (will appear in the result-path) """
        return 'COMBO'

    def exp_path(self) -> str:
        return os.path.join(super(COMBOExp, self).exp_path(), str(self.seed))

    def exp_id(self) -> str:
        eid = f'{self.method_id}_n-init-{self.n_initial}'
        if self.lamda is not None:
            eid += f'_lamda-{self.lamda:g}'
        return eid

    def run(self, verbose: bool = False):
        raise NotImplementedError()

    def evaluate(self, x: Tensor) -> Tensor:
        """

        Args:
            x: new point to evaluate
        """
        assert x.numel() == len(self.n_vertices)
        self.n_evals += 1
        # self.log(f"{self.n_evals:3d}. Evaluate sequence {x} on design {self.design_name}: ", end="")
        if x.dim() == 2:
            x = x.flatten()
        X = x.detach().cpu().numpy().astype(int)

        self.samples_X.append(X)

        sequence = [self.action_space[ind].act_str for ind in X]
        obj_1, obj_2 = get_design_prop(seq=sequence, design_file=self.design_file, mapping=self.mapping,
                                       library_file=self.library_file, abc_binary=self.abc_binary,
                                       lut_inputs=self.lut_inputs)
        self.obj_1_s.append(obj_1 / self.ref_1)
        self.obj_2_s.append(obj_2 / self.ref_2)
        return torch.tensor([obj_1 / self.ref_1 + obj_2 / self.ref_2]).to(x)

    def build_res(self, verbose: bool = False) -> COMBORes:
        objs = np.stack([self.obj_1_s, self.obj_2_s]).T
        self.samples_X = np.array(self.samples_X)
        mask_pareto = is_pareto_efficient(objs)
        history_x = self.samples_X.copy()
        history_f = objs.copy()
        samples = self.samples_X[mask_pareto]
        objs = objs[mask_pareto]
        res = COMBORes(X=samples, F=objs, history_x=history_x, history_f=history_f)
        self.exec_time = time.time() - self.ref_time
        if verbose:
            self.log(f"Took {self.exec_time} to optimise {self.design_name}")
        return res

    def process_results(self, res: COMBORes) -> pd.DataFrame:
        seq_id = []
        obj_1 = []
        ratio_1 = []
        obj_2 = []
        ratio_2 = []
        for seq_ind, func_value in zip(res.X, res.F):
            seq_id.append(' | '.join([self.action_space[ind].act_id for ind in seq_ind]))
            ratio_1.append(func_value[0])
            ratio_2.append(func_value[1])
            obj_1.append(ratio_1[-1] * self.ref_1)
            obj_2.append(ratio_2[-1] * self.ref_2)
        pd_res = pd.DataFrame()
        pd_res['seq_id'] = seq_id

        pd_res[self.obj1_id] = obj_1
        pd_res[self.obj2_id] = obj_2

        pd_res['ratio ' + self.obj1_id] = ratio_1
        pd_res['ratio ' + self.obj2_id] = ratio_2

        pd_res['qor'] = np.array(ratio_1) + np.array(ratio_2)

        return pd_res.sort_values('qor')

    def save_results(self, res: COMBORes) -> None:
        save_path = self.exp_path()
        log(
            f'{self.exp_id()} -> Save to {save_path} | Best QoR improvement over {self.ref_abc_seq}:'
            f' {(2 - res.F.sum(-1).min()) / 2 * 100:.2f}')
        os.makedirs(save_path, exist_ok=True)

        # save table of results
        pd_res = self.process_results(res)
        res_path = os.path.join(save_path, 'res.csv')
        pd_res.to_csv(res_path)

        # save execution time
        np.save(os.path.join(save_path, 'exec_time.npy'), np.array(self.exec_time))

        # save config
        save_w_pickle(self.get_config(), save_path, filename='config.pkl')

        # save res
        save_w_pickle(res, save_path, 'res.pkl')
        log(f'{self.exp_id()} -> Saved!')

    def exists(self) -> bool:
        """ Check if experiment already exists """
        save_path = self.exp_path()
        paths_to_check = [
            os.path.join(save_path, 'res.csv'),
            os.path.join(save_path, 'exec_time.npy'),
            os.path.join(save_path, 'config.pkl'),
            os.path.join(save_path, 'res.pkl')
        ]
        return np.all(list(map(lambda p: os.path.exists(p), paths_to_check)))

    @staticmethod
    def get_history_values(res: COMBORes) -> Tuple[np.ndarray, np.ndarray]:
        return get_history_values_from_res(res)

    def log(self, msg: str, end=None) -> None:
        header = f'{self.method_id} | {self.design_name}'
        log(msg, header=header, end=end)
