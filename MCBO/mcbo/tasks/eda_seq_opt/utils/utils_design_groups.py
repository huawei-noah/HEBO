# Comes from BOiLS
import glob
import os
from typing import List, Dict

import numpy as np

from .utils import get_circuits_path_root

EPFL_ARITHMETIC = ['hyp', 'div', 'log2', 'multiplier', 'sqrt', 'square', 'sin', 'bar', 'adder', 'max']
EPFL_CONTROL = ['arbiter', 'cavlc', 'ctrl', 'dec', 'i2c', 'int2float', 'mem_ctrl', 'priority', 'router', 'voter']
EPFL_MTM = ['sixteen', 'twenty', 'twentythree']

OPEN_ABC_ORIG = ['i2c_orig', 'spi_orig', 'des3_area_orig', 'ss_pcm_orig', 'usb_phy_orig', 'sasc_orig', 'wb_dma_orig',
                 'simple_spi_orig', 'dynamic_node_orig', 'aes_orig', 'pci_orig', 'ac97_ctrl_orig', 'mem_ctrl_orig',
                 'tv80_orig', 'fpu_orig', 'wb_conmax_orig', 'tinyRocket_orig', 'aes_xcrypt_orig', 'aes_secworks_orig',
                 'jpeg_orig', 'bp_be_orig', 'ethernet_orig', 'vga_lcd_orig', 'picosoc_orig', 'dft_orig', 'idft_orig',
                 'fir_orig', 'iir_orig', 'sha256_orig']

OPEN_SOURCE = ['dft', 'idft', 'i2c', 'router', 'dec', 'int2float', 'priority', 'mem_ctrl', 'arbiter', 'ctrl', 'voter',
               'cavlc', 's991',
               'c432', 'c880', 'c2670', 's499', 's3384', 's1512', 's938', 'c1908', 's3271', 's3330', 's1269', 's967',
               'c3540', 'c6288', 'c499', 's6669', 'c5315', 's4863', 'c7552', 'prolog', 's9234', 'c1355', 's635', 'b05',
               'b03_opt', 'b19', 'b05_opt_C', 'b13', 'b12', 'b21_C', 'b06', 'b08', 'b04_opt_C', 'b20_C', 'b06_C',
               'b07_opt', 'b22_C', 'b09', 'b03_C', 'b13_opt', 'b02', 'b14', 'b17_1_C', 'b04_opt', 'b05_opt', 'b03',
               'b11_C', 'b08_opt_C', 'b13_opt_C', 'b12_C', 'b09_opt_C', 'b13_C', 'b01_C', 'b03_opt_C', 'b10_opt',
               'b01_opt', 'b04', 'b01_opt_C', 'b09_opt', 'b06_opt', 'b06_opt_C', 'b12_opt', 'b01', 'b09_C', 'b12_opt_C',
               'b10_opt_C', 'b07', 'b10', 'b11_opt_C', 'b02_opt_C', 'b02_opt', 'b04_C', 'b15_1', 'b08_opt', 'b07_opt_C',
               'b11_opt', 'b11', 'b05_C', 'b08_C', 'b07_C', 'b10_C', 'b02_C', 'spi', 's208', 'des_perf', 'ac97_ctrl',
               'tv80', 'leon3', 's38584', 's641', 'simple_spi', 'ss_pcm', 'pci_bridge32', 's953', 's510', 's35932',
               's13207', 's420', 's526', 's713', 'sasc', 's832', 's15850', 's349', 's298', 's444', 'leon3mp',
               'steppermotordrive', 's382', 's1196', 's38417', 's526n', 'usb_funct', 's1488', 's344', 's1494', 's838',
               's1238', 'aes_core', 'systemcaes', 's27', 'pci_spoci_ctrl', 'usb_phy', 's820', 's400', 'des_area',
               'vga_lcd', 'pci_conf_cyc_addr_dec', 'netcard', 'wb_conmax', 'leon2', 'ethernet', 's386', 's1423',
               'wb_dma', 'systemcdes', 's5378', 'g125', 'g25', 'g36', 'g216', 'LEKU-CB', 'g625', 'g1296', 'LEKU-CD',
               'aes_xcrypt', 'dynamic_node', 'aes', 'fpu', 'iir', 'picosoc', 'mainpla', 'jpeg', 'bc0', 'des3_area',
               'apex1', 'bp_be', 'aes_secworks', 'fir', 'dalu', 'sha256', 'k2', 'tinyRocket', 'pci', 'i10',
               'div', 'log2', 'multiplier', 'sqrt', 'square', 'sin', 'bar', 'adder', 'max', 'hyp']

DESIGN_GROUPS: Dict[str, List[str]] = {
    'epfl_arithmetic': EPFL_ARITHMETIC,
    'epfl_control': EPFL_CONTROL,
    'epfl_mtm': EPFL_MTM,
    'open_source': OPEN_SOURCE,
    'open_abc_orig': OPEN_ABC_ORIG
}

for file in glob.glob(f"{get_circuits_path_root()}/*.blif"):
    circ_name = os.path.basename(file[:-5])
    DESIGN_GROUPS[circ_name] = [circ_name]

AUX_TEST_GP = ['adder', 'bar']
AUX_TEST_ABC_GRAPH = ['adder', 'sin']

DESIGN_GROUPS['aux_test_designs_group'] = AUX_TEST_GP
DESIGN_GROUPS['aux_test_abc_graph'] = AUX_TEST_ABC_GRAPH


def get_designs_path(designs_id: str, frac_part: str = None) -> List[str]:
    """ Get list of filepaths to designs """

    designs_filepath: List[str] = []
    if designs_id in DESIGN_GROUPS:
        group = DESIGN_GROUPS[designs_id]
    else:
        try:
            from mcbo.tasks.eda_seq_opt.utils.utils_design_groups__perso__ import DESIGN_GROUPS_PERSO
            if designs_id in DESIGN_GROUPS_PERSO:
                group = DESIGN_GROUPS_PERSO[designs_id]
        except ModuleNotFoundError:
            pass
    for design_id in group:
        designs_filepath.append(os.path.join(get_circuits_path_root(), f'{design_id}.blif'))
    if frac_part is None:
        s = slice(0, len(designs_filepath))
    else:
        i, j = map(int, frac_part.split('/'))
        assert j > 0 and i > 0, (i, j)
        step = int(np.ceil(len(designs_filepath) / j))
        s = slice((i - 1) * step, i * step)

    return designs_filepath[s]


if __name__ == '__main__':

    designs_id_ = 'test_designs_group'
    N = 6
    for n in range(1, N + 1):
        frac = f'{n}/{N}'
        print(f'{frac} -----> ', end='')
        print(get_designs_path(designs_id=designs_id_, frac_part=frac))
