# 2021.11.10-Move abc import
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import sys
import time
import traceback
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from utils.utils_save import save_w_pickle, get_storage_root, get_storage_data_root, load_w_pickle
from utils.utils_misc import time_formatter, log

from resources.abcRL.graphExtractor import extract_dgl_graph

designs = ['a8091', 'adder', 'alu4', 'apex2', 'apex4', 'arbiter', 'b0032', 'b1209', 'b1328', 'b2098', 'b2129',
           'b2138', 'b2640', 'b4196', 'b7317', 'b8272', 'b8940', 'b9833', 'bar', 'bigkey', 'c0103',
           'c0762', 'c3760', 'c6691', 'c7805', 'cam', 'cavlc', 'cic', 'clma', 'ctrl', 'dec', 'des', 'diffeq', 'div',
           'dsip', 'elliptic', 'ex1010', 'ex5p', 'frisc', 'gpio', 'hyp', 'i2c', 'ics1_c1355', 'ics1_c17', 'ics1_c1908',
           'ics1_c2670', 'ics1_c3540', 'ics1_c432', 'ics1_c499', 'ics1_c5315', 'ics1_c6288', 'ics1_c880', 'ics1_s1196',
           'ics1_s1238', 'ics1_s13207', 'ics1_s1423', 'ics1_s1488', 'ics1_s1494', 'ics1_s15850', 'ics1_s27',
           'ics1_s298_ORIG', 'ics1_s344', 'ics1_s349',
           'ics1_s35932', 'ics1_s382', 'ics1_s38417', 'ics1_s38584', 'ics1_s386', 'ics1_s400', 'ics1_s420', 'ics1_s444',
           'ics1_s510', 'ics1_s526', 'ics1_s526n', 'ics1_s641', 'ics1_s713', 'ics1_s820', 'ics1_s832', 'ics1_s838',
           'ics1_s9234', 'ics1_s953', 'int2float',
           'log2', 'max', 'mem', 'misex3', 'multiplier',
           'pdc', 'priority', 'router', 's298', 's38417', 's38584', 'sddac', 'seq', 'sin', 'spla', 'sqrt', 'square',
           'tseng', 'voter']

segfault_designs = ['b0279', 'b8391', 'c0400', 'c2593', 'c3738', 'c3961', 'c4618', 'c5927', 'c6500', 'ics1_s5378',
                    'list']

mtm_designs = ['twentythree', 'twenty', 'sixteen']


def resyn(abc_instance):
    abc_instance.balance(l=False)
    abc_instance.rewrite(l=False)
    abc_instance.refactor(l=False)
    abc_instance.balance(l=False)
    abc_instance.rewrite(l=False)
    abc_instance.rewrite(l=False, z=True)
    abc_instance.balance(l=False)
    abc_instance.refactor(l=False, z=True)
    abc_instance.rewrite(l=False, z=True)
    abc_instance.balance(l=False)


def get_path(design_instance: str) -> str:
    return os.path.join(get_storage_root(), 'time_test', design_instance)


def already_exists(design_instance: str) -> bool:
    """ Check if time checker has already been run for the given design """
    design_path = get_path(design_instance)
    result_path = os.path.join(design_path, 'results.pkl')
    if not os.path.exists(result_path):
        return False
    results = load_w_pickle(design_path, 'results.pkl')
    for k in ['time_resyn2', 'time_stats', 'time_extract_graph', 'time_read']:
        if k not in results:
            return False
    return True


if __name__ == '__main__':
    import abc_py

    abc = abc_py.AbcInterface()

    for design in designs:
        if already_exists(design_instance=design):
            log(f'{design}: Already have results -> SKIP IT')
            continue
        abc.start()
        log(f'Start {design}')
        result = {}
        design_file = os.path.join(get_storage_data_root(), f'benchmark_blif/{design}.blif')

        try:
            time_ref = time.time()
            abc.read(design_file)
            result['time_read'] = time.time() - time_ref
            log(f"{design}: time_read -> {time_formatter(result['time_read'], show_ms=True)}")

            # extract graph
            time_ref = time.time()
            G = extract_dgl_graph(abc)
            result['time_extract_graph'] = time.time() - time_ref
            log(f"{design}: time_extract_graph -> {time_formatter(result['time_extract_graph'], show_ms=True)}")

            time_ref = time.time()
            stats = abc.aigStats()
            result['time_stats'] = time.time() - time_ref
            log(f"{design}: time_stats -> {time_formatter(result['time_stats'], show_ms=True)}")

            result['nodes'] = stats.numAnd
            result['levels'] = stats.lev
            log(f"{design}: num_nodes -> {result['nodes']}")
            log(f"{design}: num_levels -> {result['levels']}")

            # resyn2
            time_ref = time.time()
            resyn(abc)
            result['time_resyn2'] = time.time() - time_ref
            log(f"{design}: time_resyn2 -> {time_formatter(result['time_resyn2'], show_ms=True)}")
        except Exception as e:
            result['exc'] = traceback.format_exc()
            log(f"{design} -> {result['exc']}")

        os.makedirs(get_path(design_instance=design), exist_ok=True)
        save_w_pickle(result, get_path(design_instance=design), 'results.pkl')
        log(f"{design} -> save results in {get_path(design_instance=design)}")
        abc.end()
