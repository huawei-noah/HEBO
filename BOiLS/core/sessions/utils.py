# 2021.11.10-updated the metrics retrieved from the stats
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import re
from subprocess import check_output
from typing import List, Optional, Tuple, Dict, Union, Any

from utils.utils_misc import log
from core.sessions.utils_eval import fpga_evaluate


def get_metrics(stats) -> Dict[str, Union[float, int]]:
    """
    parse LUT count and levels from the stats command of ABC
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()

    results = {}
    ob = re.search(r'lev *= *[0-9]+', line)
    results['levels'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'nd *= *[0-9]+', line)
    results['lut'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
    results['input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
    results['output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())

    ob = re.search(r'edge *= *[0-9]+', line)
    results['edges'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'lat *= *[0-9]+', line)
    results['latches'] = int(ob.group().split('=')[1].strip())

    return results


def get_fpga_design_prop(library_file: str, design_file: str, abc_binary: str, lut_inputs: int,
                         sequence: List[str] = None, write_unmap_design_path: Optional[str] = None,
                         verbose: Optional[int] = 0) -> Tuple[int, int]:
    """
    Compute and return lut_k and levels associated to a specific design

    Args:
        library_file: standard cell library mapping
        design_file: path to the design file
        abc_binary: abc binary path
        sequence: sequence of operations (containing final ';') to apply to the design
        lut_inputs: number of LUT inputs (2 < num < 33)
        verbose: verbosity level
        write_unmap_design_path: path where to store the design obtained after the sequence of actions have been applied

    Returns:
        lut_K, levels
    """
    if sequence is None:
        sequence = ['strash; ']
    if 'strash' not in sequence[0]:
        sequence.insert(0, 'strash; ')
    abc_command = 'read ' + library_file + '; '
    abc_command += 'read ' + design_file + '; '
    abc_command += ' '.join(sequence)
    if write_unmap_design_path is not None:
        abc_command += f"write {write_unmap_design_path}; "
    abc_command += f"if {'-v ' if verbose > 0 else ''}-K {lut_inputs}; "
    abc_command += 'print_stats; '
    print(f"{abc_binary} -c '{abc_command}'")
    if verbose:
        log(abc_command)
    proc = check_output([abc_binary, '-c', abc_command])
    results = get_metrics(proc)
    print(results['lut'], results['levels'])
    return results['lut'], results['levels']


def get_design_prop(seq: List[str], design_file: str, mapping: str, library_file: str,
                    abc_binary: str, lut_inputs: int, use_yosys: bool, compute_init_stats: bool, verbose: bool = False,
                    write_unmap_design_path: Optional[str] = None) -> Tuple[int, int, Dict[str, Any]]:
    """
     Get property of the design after applying sequence of operations

    Args:
        seq: sequence of operations
        design_file: path to the design
        mapping: either scl of fpga mapping
        library_file: library file (asap7.lib)
        abc_binary: (probably yosys-abc)
        lut_inputs: number of LUT inputs (2 < num < 33)
        verbose: verbosity level
        compute_init_stats: whether to compute and store initial stats
        write_unmap_design_path: path where to store the design obtained after the sequence of actions have been applied

    Returns:
        either:
            - for fpga: lut_k, level
            - for scl: area, delay
    """

    if mapping == 'fpga':
        # lut_k, levels = get_fpga_design_prop(
        #     library_file=library_file,
        #     design_file=design_file,
        #     abc_binary=abc_binary,
        #     lut_inputs=lut_inputs,
        #     sequence=seq,
        #     verbose=verbose,
        #     write_unmap_design_path=write_unmap_design_path
        # )
        # assert not write_unmap_design_path, "[Deprecated] Does not support this option anymore"
        lut_k, levels, extra_info = fpga_evaluate(design_file=design_file, sequence=seq, lut_inputs=lut_inputs,
                                                  compute_init_stats=compute_init_stats, verbose=verbose,
                                                  use_yosys=use_yosys, write_unmap_design_path=write_unmap_design_path)
        return lut_k, levels, extra_info
    elif mapping == 'scl':
        raise NotImplementedError()
    else:
        raise ValueError(mapping)
