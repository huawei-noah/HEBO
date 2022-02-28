# 2021.11.10-add support to new actions
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)  # should points to the root of the project
sys.path[0] = ROOT_PROJECT

from utils.utils_cmd import parse_list, parse_dict
from typing import List, Union, Tuple, Dict, Any, Optional
import time


def fpga_evaluate_aux(design_file: str, sequence: List[Union[str, int]],
                      compute_init_stats: bool, lut_inputs: int, verbose: bool = False) \
        -> Dict[str, Any]:
    """
         Get property of the design after applying sequence of operations `sequence`

        Args:
            design_file: path to the design 'path/to/design.blif'
            sequence: sequence of operations
                        -> either identified by id number
                            0: rewrite
                            1: rewrite -z...
                        -> or by operation names
                            `rewrite`
                            `rewrite -z`
                            ...
            lut_inputs: number of LUT inputs (2 < num < 33)
            compute_init_stats: whether to compute and store initial stats on delay and area
            verbose: verbosity level
        Returns:
            dictionary with:
                lut
                levels
                extra_info (execution time, initial stats)
    """
    import abc_py

    t_ref = time.time()
    abc = abc_py.AbcInterface()
    abc.start()
    abc.read(design_file)

    extra_info: Dict[str, Any] = {}

    if compute_init_stats:
        abc.map(v=verbose, k=lut_inputs)
        stats = abc.aigStatsMapping()
        extra_info['init_levels'] = stats.lev
        extra_info['init_area'] = stats.numAnd

    abc.strash()

    for action in sequence:
        print(action, time.time() - t_ref)
        if action == 0 or action == 'rewrite':
            abc.rewrite()  # rw
        elif action == 1 or action == 'rewrite -z':
            abc.rewrite(z=True)  # rw -z
        elif action == 2 or action == 'refactor':
            abc.refactor()  # rf
        elif action == 3 or action == 'refactor -z':
            abc.refactor(z=True)  # rf -z
        elif action == 4 or action == 'resub':
            abc.resub()  # rs
        elif action == 5 or action == 'resub -z':
            abc.resub(z=True)  # rs -z
        elif action == 6 or action == 'balance':
            abc.balance()  # b
        elif action == 7 or action == 'fraig':
            abc.fraig()  # f
        elif action == 8 or action == '&sopb':
            abc.sopb()
        elif action == 9 or action == '&blut':
            abc.blut()
        elif action == 10 or action == '&dsdb':
            abc.dsdb()
        else:
            raise NotImplementedError(action)
    abc.map(v=verbose, k=lut_inputs)
    stats = abc.aigStatsMapping()
    abc.end()
    extra_info['exec_time'] = time.time() - t_ref
    return dict(levels=stats.lev, lut=stats.numAnd, extra_info=extra_info)


def fpga_evaluate(design_file: str, sequence: List[Union[str, int]], lut_inputs: int, use_yosys: bool,
                  compute_init_stats: bool = False, verbose: bool = False,
                  write_unmap_design_path: Optional[str] = None) \
        -> Tuple[int, int, Dict[str, Any]]:
    """
         Get property of the design after applying sequence of operations `sequence`

        Args:
            design_file: path to the design 'path/to/design.blif'
            sequence: sequence of operations
                        -> either identified by id number
                            0: rewrite
                            1: rewrite -z...
                        -> or by operation names
                            `rewrite`
                            `rewrite -z`
                            ...
            lut_inputs: number of LUT inputs (2 < num < 33)
            use_yosys: whether to use yosys-abc or abc_py
            compute_init_stats: whether to compute and store initial stats on delay and area
            verbose: verbosity level
        Returns:
            lut_k, level and extra_info (execution time, initial stats)
        Exception: CalledProcessError
    """
    if not use_yosys:
        assert write_unmap_design_path is None
        stats = subprocess.check_output([
            "python3", f"{ROOT_PROJECT}/core/sessions/utils_eval.py",
            "--design_file", design_file,
            '--actions', f"\"{str(sequence)}\"",
            '--lut_inputs', str(lut_inputs),
            '--compute_init_stats', str(int(compute_init_stats)),
            '--verbose', str(int(verbose))
        ], stderr=subprocess.STDOUT)
        print(stats)
        results = parse_dict(stats.decode("utf-8").split('\n')[-2])
        print(results)
        return results['lut'], results['levels'], results['extra_info']

    assert not compute_init_stats
    t_ref = time.time()
    extra_info: Dict[str, Any] = {}

    if sequence is None:
        sequence = []
    sequence = ['strash; '] + sequence
    abc_command = 'read ' + design_file + '; '
    abc_command += ';'.join(sequence) + '; '
    if write_unmap_design_path is not None:
        abc_command += 'write ' + write_unmap_design_path + '; '
    abc_command += f"if {'-v ' if verbose > 0 else ''}-K {lut_inputs}; "
    abc_command += 'print_stats; '
    cmd_elements = ['yosys-abc', '-c', abc_command]
    proc = subprocess.check_output(cmd_elements)
    # read results and extract information
    line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()

    ob = re.search(r'lev *= *[0-9]+', line)
    if ob is None:
        print("----" * 10)
        print(f'Command: {" ".join(cmd_elements)}')
        print(f"Out line: {line}")
        print(f"Design: {design_file}")
        print(f"Sequence: {sequence}")
        print("----" * 10)
    levels = int(ob.group().split('=')[1].strip())

    ob = re.search(r'nd *= *[0-9]+', line)
    lut = int(ob.group().split('=')[1].strip())

    extra_info['exec_time'] = time.time() - t_ref
    return lut, levels, extra_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.add_argument('--design_file', type=str, help='path to blif design')
    parser.add_argument('--actions', type=list, help='Sequence of actions')
    parser.add_argument('--lut_inputs', type=int)
    parser.add_argument('--compute_init_stats', type=int, default=0)

    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    assert args.lut_inputs > 0, args.lut_inputs

    results_ = fpga_evaluate_aux(
        design_file=args.design_file,
        lut_inputs=args.lut_inputs,
        sequence=args.actions,
        compute_init_stats=args.compute_init_stats,
        verbose=args.verbose
    )
    print(results_)
