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

from core.algos.common_exp import MultiEADExp

META_METHOD_ID = 'RL'
METHOD_ID = 'GRiLLS'


def get_exp_path(mapping: str, lut_inputs: int, seq_length: int, action_space_id: str,
                 exp_id: str, design_files_id: str, ref_abc_seq: str, seed: int):
    return os.path.join(MultiEADExp.get_exp_path_aux(
        meta_method_id=META_METHOD_ID,
        mapping=mapping,
        lut_inputs=lut_inputs,
        seq_length=seq_length,
        action_space_id=action_space_id,
        exp_id=exp_id,
        design_files_id=design_files_id,
        ref_abc_seq=ref_abc_seq
    ), str(seed))


def get_exp_id(objective: str, use_yosys: bool, n_episodes: int, alpha_pi: float, alpha_v: float,
               gamma: float) -> str:
    exp_id = METHOD_ID
    exp_id += f"_ep-{n_episodes}"
    exp_id += f"_obj-{objective}"
    exp_id += f"_alpha-pi-{alpha_pi:g}"
    exp_id += f"_alpha-v-{alpha_v:g}"
    exp_id += f"_gamma-{gamma:g}"
    if use_yosys:
        exp_id += '_yosys'
    return exp_id
