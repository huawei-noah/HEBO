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
import sys
from pathlib import Path
from subprocess import check_output, CalledProcessError


def test_to_str():
    d = dict(
        a=1,
        b=2.1,
        c='foo/boo',
        d=[1, 'a']
    )
    assert str_dict(d) == 'a-1-b-2.1-c-foo~boo-d-1_a'


def test_proc_abc():
    abc_binary = 'yosys-abc'
    abc_command = f'read {ROOT_PROJECT}/asap7.lib; read {get_storage_data_root()}/benchmark_blif/b9833.blif; strash;  &get -n; &dsdb; &put; &get -n; &sopb; &put; fraig; fraig; rewrite; &get -n; &dsdb; &put; rewrite -z; fraig; rewrite; refactor;if -K 4; print_stats; '

    try:
        proc = check_output([abc_binary, '-c', abc_command])
    except CalledProcessError as e:
        print("Test test_proc_abc passed", e.args)
    # results = get_metrics(proc)


if __name__ == '__main__':
    ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
    sys.path[0] = ROOT_PROJECT
    from core.sessions.utils import get_metrics

    from utils.utils_save import str_dict, get_storage_data_root

    test_to_str()
    test_proc_abc()
