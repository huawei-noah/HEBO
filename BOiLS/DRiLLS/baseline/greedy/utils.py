import os
import subprocess
import re


def extract_results(stats):
    """
    extracts area and delay from the printed stats on stdout
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()

    ob = re.search(r'Delay *= *[1-9]+.?[0-9]*', line)
    delay = float(ob.group().split('=')[1].strip())
    ob = re.search(r'Area *= *[1-9]+.?[0-9]*', line)
    area = float(ob.group().split('=')[1].strip())
    return delay, area


def run_optimization(output_dir, optimization, design_file, library, clock_period: float):
    """
    returns new_design_file, delay, area
    """
    output_dir = output_dir.replace(' ', '_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_design_file = output_dir + '/design.blif'

    abc_command = 'read ' + library + '; '
    abc_command += 'read ' + design_file + '; '
    abc_command += 'strash; '
    abc_command += optimization + '; '
    abc_command += 'write ' + output_design_file + '; '
    abc_command += 'map -D ' + str(clock_period) + '; '
    abc_command += 'topo; stime; '

    proc = subprocess.check_output(['yosys-abc', '-c', abc_command])
    d, a = extract_results(proc)
    return output_design_file, d, a


def run_post_mapping(output_dir, optimization, design_file, library, clock_period: float):
    """
    returns new_design_file, delay, area
    """
    output_dir = output_dir.replace(' ', '_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_design_file = output_dir + '/design.blif'

    abc_command = 'read ' + library + '; '
    abc_command += 'read ' + design_file + '; '
    abc_command += 'strash; '
    abc_command += 'map -D ' + str(clock_period) + '; '
    abc_command += optimization + ';'
    abc_command += 'write ' + output_design_file + '; '
    abc_command += 'print_stats; '
    proc = subprocess.check_output(['yosys-abc', '-c', abc_command])
    d, a = extract_results(proc)
    return output_design_file, d, a


def run_thread(iteration_dir, design_file, opt, library_file, clock_period: float, log=None):
    opt_dir = os.path.join(iteration_dir, opt)
    opt_file, delay, area = run_optimization(opt_dir, opt,
                                             design_file,
                                             library=library_file, clock_period=clock_period)
    if log is not None:
        log('Optimization: ' + opt + ' -> delay: ' + str(delay) + ', area: ' + str(area))
    return opt, opt_file, delay, area


def run_thread_post_mapping(iteration_dir, design_file, opt, library_file, clock_period, log=None):
    opt_dir = os.path.join(iteration_dir, opt)
    opt_file, delay, area = run_post_mapping(opt_dir, opt,
                                             design_file,
                                             library=library_file, clock_period=clock_period)
    if log is not None:
        log('Optimization: ' + opt + ' -> delay: ' + str(delay) + ', area: ' + str(area))
    return opt, opt_file, delay, area
