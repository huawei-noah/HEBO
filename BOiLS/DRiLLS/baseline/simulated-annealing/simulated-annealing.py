#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import yaml
import os
import subprocess
import sys
import timeit
import re
import random
import math
from joblib import Parallel, delayed

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    options = yaml.load(f)

start = timeit.default_timer()

optimizations = options['optimizations']
iterations = options['iterations']
current_design_file = options['design_file']
library_file = options['mapping']['library_file']
clock_period = options['mapping']['clock_period']
# post_mapping_optimizations = options['post_mapping_commands']

temperature = options['simulated_annealing']['initial_temp']
cooling_rate = options['simulated_annealing']['cooling_rate']

# Create directory if not exists
if not os.path.exists(options['output_dir']):
    os.makedirs(options['output_dir'])

def extract_results(stats):
    """
    extracts area and delay from the printed stats on stdout
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    ob = re.search(r'delay *= *[1-9]+.?[0-9]+', line)
    delay = float(ob.group().split('=')[1].strip())
    ob = re.search(r'area *= *[1-9]+.?[0-9]+', line)
    area = float(ob.group().split('=')[1].strip())
    return delay, area

def run_optimization(output_dir, optimization, design_file, library):
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
    abc_command += 'print_stats; '
    
    proc = subprocess.check_output(['yosys-abc','-c', abc_command])
    d, a = extract_results(proc)
    return output_design_file, d, a

def save_optimization_step(iteration, optimization, delay, area):
    """
    saves the winning optimization to a csv file
    """
    with open(os.path.join(options['output_dir'], 'results.csv'), 'a') as f:
        data_point = str(iteration) + ', ' + str(optimization) + ', '
        data_point += str(delay) + ', ' + str(area) + '\n'
        f.write(data_point)

def log(message=''):
    print(message)
    with open(os.path.join(options['output_dir'], 'greedy.log'), 'a') as f:
        f.write(message + '\n')
    
def run_post_mapping(output_dir, optimization, design_file, library):
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
    proc = subprocess.check_output(['yosys-abc','-c', abc_command])
    d, a = extract_results(proc)
    return output_design_file, d, a

def run_thread(iteration_dir, design_file, opt):
    opt_dir = os.path.join(iteration_dir, opt)
    opt_file, delay, area = run_optimization(opt_dir, opt, 
                                                     design_file, 
                                                     library_file)
    log('Optimization: ' + opt + ' -> delay: ' + str(delay) + ', area: ' + str(area))
    return (opt, opt_file, delay, area)

def run_thread_post_mapping(iteration_dir, design_file, opt):
    opt_dir = os.path.join(iteration_dir, opt)
    opt_file, delay, area = run_post_mapping(opt_dir, opt, 
                                                     design_file, 
                                                     library_file)
    log('Optimization: ' + opt + ' -> delay: ' + str(delay) + ', area: ' + str(area))
    return (opt, opt_file, delay, area)

i = 0
# run the optimization once to set the initial energy (delay) of the system
log('Initializing annealing ..')
log('Current temperature: ' + str(temperature))
log('----------------')
iteration_dir = os.path.join(options['output_dir'], str(i))
if not os.path.exists(iteration_dir):
    os.makedirs(iteration_dir)
# Pick an optimization at random
random_optimization = 'strash'      # a command that does no optimization
result = run_thread(iteration_dir, current_design_file, random_optimization)
opt_file = result[1]
delay = result[2]
area = result[3]
# accept it to set the energe of the system in the beginning
save_optimization_step(i, random_optimization, delay, area)
current_design_file = opt_file
previous_delay = delay
i += 1

log('System initialized with delay: ' + str(delay))
log('Starting annealing ..')
log()

# main optimizing iteration
while True:
    number_of_accepted_optimizations = 0

    for _ in range(100):
        # if we accept 10 optimizations, we cool down the system
        # otherwise, only continue up to 100 trials for this temperature
        
        # log
        log('Iteration: ' + str(i))
        log('Temperature: ' + str(temperature))
        log('----------------')
    
        # create a directory for this iteration
        iteration_dir = os.path.join(options['output_dir'], str(i))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
    
        # Pick an optimization at random
        random_optimization = random.choice(optimizations)
        result = run_thread(iteration_dir, current_design_file, random_optimization)
        opt_file = result[1]
        delay = result[2]
        area = result[3]

        # if better than the previous delay, accept. Otherwise, accept with probability
        if delay < previous_delay:
            log('The optimization reduced the delay!')
            log('Accepting it ..')
            save_optimization_step(i, random_optimization, delay, area)
            current_design_file = opt_file
            previous_delay = delay
            number_of_accepted_optimizations += 1
        else:
            delta_delay = delay - previous_delay
            probability_of_acceptance = math.exp((- delta_delay) / temperature)
            log('The optimization didn\'t reduce the delay, the system looks to be still hot.')
            log('The probability of acceptance is: ' + str(probability_of_acceptance))
            log('Uniformly generating a number to see if we accept it ..')
            if random.uniform(0, 1.0) < probability_of_acceptance:
                log('Accepting it ..')
                save_optimization_step(i, random_optimization, delay, area)
                current_design_file = opt_file
                previous_delay = delay
                number_of_accepted_optimizations += 1
            else:
                log('Rejected ..')
                pass
        i += 1
        log()

        if number_of_accepted_optimizations == 10:
            break

    if temperature <= 0.1:
        log('System has sufficiently cooled down ..')
        log('Shutting down simulation ..')
        log()
        break

    new_temperature = temperature * cooling_rate
    log('Cooling down system from ' + str(temperature) + ' to ' + str(new_temperature) + ' ..')
    temperature = new_temperature
    log('================')
    log()

stop = timeit.default_timer()

log('Total Optimization Time: ' + str(stop - start))
