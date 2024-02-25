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

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

import random
import time

import numpy as np
import torch

from mcbo.task_factory import task_factory
from mcbo.optimizers.non_bo.genetic_algorithm import PymooGeneticAlgorithm, CategoricalGeneticAlgorithm


def print_results(our_results, our_time, pymoo_results, pymoo_time, pop_size, num_iter, num_dims, num_categories,
                  tournament_selection, random_seeds):
    print(f'Pymoo Tournament selection: {tournament_selection} - Number of random seeds: {len(random_seeds)}')
    for task_name in our_results:
        print(
            f'\nTask name : {task_name} (Num dims: {num_dims}, num categories per dim: {num_categories}, GA pop_size: {pop_size}, num iter: {num_iter})\n')
        print(
            f'Our GA   - Found minima: ({np.mean(our_results[task_name]):.2f}+-{np.std(our_results[task_name]):.2f}) - Time taken: ({np.mean(our_time[task_name]):.2f}+-{np.std(our_time[task_name]):.2f})s')
        print(
            f'Pymoo GA - Found minima: ({np.mean(pymoo_results[task_name]):.2f}+-{np.std(pymoo_results[task_name]):.2f}) - Time taken: ({np.mean(pymoo_time[task_name]):.2f}+-{np.std(pymoo_time[task_name]):.2f})s')
        print()


if __name__ == '__main__':

    tournament_selection = True
    pop_size = 40
    random_seeds = [42, 43, 44, 45, 46]
    task_names = ['ackley', 'griewank', 'langermann', 'levy', 'rastrigin', 'schwefel']
    num_iter = 2000
    num_dims = 20
    num_categories = 11

    pymoo_results = {}
    pymoo_time = {}
    our_results = {}
    our_time = {}

    for task_name in task_names:
        print(f'Task {task_name}')
        pymoo_results[task_name] = []
        pymoo_time[task_name] = []
        our_results[task_name] = []
        our_time[task_name] = []

        for seed in random_seeds:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            task = task_factory(task_name, num_dims=num_dims, variable_type='nominal',
                                              num_categories=num_categories)
            search_space = task.get_search_space()

            x_init = search_space.sample(pop_size)

            pymoo_optimizer = PymooGeneticAlgorithm(
                search_space=search_space,
                input_constraints=None,
                obj_dims=[0],
                out_constr_dims=None,
                out_upper_constr_vals=None,
                pop_size=pop_size,
                tournament_selection=tournament_selection
            )
            pymoo_optimizer.set_x_init(x_init)

            our_optimizer = CategoricalGeneticAlgorithm(
                search_space=search_space,
                obj_dims=[0],
                out_constr_dims=None,
                out_upper_constr_vals=None,
                input_constraints=None
            )
            our_optimizer.set_x_init(x_init)

            start = time.time()
            for i in range(1, num_iter + 1):
                x_next = pymoo_optimizer.suggest()
                y_next = task(x_next)
                pymoo_optimizer.observe(x_next, y_next)
            end = time.time()

            pymoo_results[task_name].append(pymoo_optimizer.best_y)
            pymoo_time[task_name].append(end - start)

            start = time.time()
            for i in range(1, num_iter + 1):
                x_next = our_optimizer.suggest()
                y_next = task(x_next)
                our_optimizer.observe(x_next, y_next)
            end = time.time()

            our_results[task_name].append(our_optimizer.best_y)
            our_time[task_name].append(end - start)

    print_results(our_results, our_time, pymoo_results, pymoo_time, pop_size, num_iter, num_dims, num_categories,
                  tournament_selection, random_seeds)
