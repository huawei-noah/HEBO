#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

# 2021.11.10-refactored, moving run_threads to a utils script
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import sys
import timeit

import yaml
from joblib import Parallel, delayed

from DRiLLS.baseline.greedy.utils import run_thread

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    options = yaml.load(f)

start = timeit.default_timer()

optimizations = options['optimizations']
iterations = options['iterations']
current_design_file = options['design_file']
library_file = options['mapping']['library_file']
clock_period = options['mapping']['clock_period']
post_mapping_optimizations = options['post_mapping_commands']

# Create directory if not exists
if not os.path.exists(options['output_dir']):
    os.makedirs(options['output_dir'])


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


# main optimizing iteration
previous_area = None
for i in range(iterations):
    # log
    log('Iteration: ' + str(i + 1))
    log('-------------')

    # create a directory for this iteration
    iteration_dir = os.path.join(options['output_dir'], str(i))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)

    # in parallel, run ABC on each of the optimizations we have    
    results = Parallel(n_jobs=len(optimizations))(
        delayed(run_thread)(iteration_dir, current_design_file, opt, library_file, clock_period) for opt in optimizations)

    # get the minimum result of all threads
    best_thread = min(results, key=lambda t: t[3])  # getting minimum for delay (index=2) or area (index=3)

    # hold the best result in variables
    best_optimization = best_thread[0]
    best_optimization_file = best_thread[1]
    best_delay = best_thread[2]
    best_area = best_thread[3]

    if best_area == previous_area:
        # break for now
        log('Looks like the best area is exactly the same as last iteration!')
        log('Continue anyway ..')
        log('Choosing Optimization: ' + best_optimization + ' -> delay: ' + str(best_delay) + ', area: ' + str(
            best_area))
        save_optimization_step(i, best_optimization, best_delay, best_area)

        log()

        # update design file for the next iteration
        current_design_file = best_optimization_file
        log('================')
        log()
        # continue
        #
        # log()
        # log('Looks like the best area is exactly the same as last iteration!')
        # log('Performing post mapping optimizations ..')
        # # run post mapping optimization
        # results = Parallel(n_jobs=len(post_mapping_optimizations))(
        #     delayed(run_thread_post_mapping)(iteration_dir, current_design_file, opt) for opt in
        #     post_mapping_optimizations)
        #
        # # get the minimum result of all threads
        # best_thread = min(results, key=lambda t: t[3])  # getting minimum for delay (index=2) or area (index=3)
        #
        # # hold the best result in variables
        # best_optimization = best_thread[0]
        # best_optimization_file = best_thread[1]
        # best_delay = best_thread[2]
        # best_area = best_thread[3]
        # previous_area = None
    else:
        previous_area = best_area

    # save results
    log()
    log('Choosing Optimization: ' + best_optimization + ' -> delay: ' + str(best_delay) + ', area: ' + str(best_area))
    save_optimization_step(i, best_optimization, best_delay, best_area)

    # update design file for the next iteration
    current_design_file = best_optimization_file
    log('================')
    log()

stop = timeit.default_timer()

log('Total Optimization Time: ' + str(stop - start))
