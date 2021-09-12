# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import subprocess
from ast import literal_eval
from collections import deque
from shlex import shlex
from typing import Tuple, Optional, Deque, Any


def run_command(command: str, shell: bool = False, memory: int = 0) -> Tuple[Optional[int], Deque[str]]:
    """ Run command line `command`

    Args:
        command: command line to execute
        shell: If true, the command will be executed through the shell
        memory: number of lines output in the stdout system to keep in memory

    Returns:
        - The return code associated to the command execution (should be `0` if it ran normally)
        - The last `memory` lines kept in memory (`deque`) object
    """
    print(' '.join(command.split()))
    if shell:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=shell)
    else:
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, shell=shell)
    s = ''
    memory = deque(maxlen=memory)
    while True:
        output = process.stdout.readline()
        s = output.strip()
        print(s)
        memory.append(s)
        # Do something else
        return_code: Optional[int] = process.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
                memory.append(output)
            break
    return return_code, memory


def parse_dict(raw: str) -> Any:
    """
    Helper method for parsing string-encoded <dict>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <dict> {} with exception {}'.format(raw, e))


def parse_list(raw: str) -> Any:
    """
    Helper method for parsing string-encoded <list>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <list> {} with exception {}'.format(raw, e))
