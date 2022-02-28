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


import argparse
import os

from utils.utils_save import ROOT_PROJECT


def add_common_args(parser: argparse.ArgumentParser):
    eda_group = parser.add_argument_group("EDA optimisation")
    eda_group.add_argument("--design_file", type=str, default=None, help="Design filepath")
    eda_group.add_argument("--designs_group_id", type=str, default=None, help="ID of group of designs to consider")
    eda_group.add_argument("--frac_part", type=str, default=None,
                           help="Which part of the group to consider (should follow the pattern `i/j`)")
    eda_group.add_argument("--seq_length", type=int, required=True, help="length of the optimal sequence to find")
    eda_group.add_argument("--mapping", type=str, default='fpga', choices=('fpga', 'scl'),
                           help="Map to standard cell library or FPGA")
    eda_group.add_argument("--lut_inputs", type=int, required=True, help="number of LUT inputs (2 < num < 33)")
    eda_group.add_argument("--action_space_id", type=str, default='standard',
                           help="id of action space defining avaible abc optimisation operations")
    eda_group.add_argument("--library_file", type=str, default=os.path.join(ROOT_PROJECT, 'asap7.lib'),
                           help="library file for mapping")
    eda_group.add_argument("--abc_binary", type=str, default='yosys-abc')
    eda_group.add_argument("--ref_abc_seq", type=str, default='resyn2',
                           help="sequence of operations to apply to initial design to get reference performance")
    parser.add_argument("--use_yosys", type=int, choices=(0, 1), help='whether to use yosys-abc or abc_py')

    parser.add_argument("--overwrite", action='store_true', help='Overwrite existing experiment')

    return parser
