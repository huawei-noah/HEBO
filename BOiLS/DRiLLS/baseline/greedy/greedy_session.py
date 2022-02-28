# 2021.11.10-add removal of saved .blif files
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import glob
import os
from typing import List

import numpy as np
from joblib import Parallel, delayed

from DRiLLS.baseline.greedy.utils import run_thread
from utils.utils_misc import log


class GreedySclSession:

    def __init__(self, params, design_file: str, playground_dir: str, max_iteration: int,
                 ref_delay: float, ref_area: float, metric: str):

        assert metric in ('min', 'sum', 'constraint'), metric

        self.max_iterations = max_iteration
        self.params = params
        self.playground_dir = playground_dir

        self.design_file = design_file
        self.current_design_file = design_file

        self.action_space_length = len(self.params['optimizations'])

        self.iteration = 0
        self.episode = 0
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))

        self.sequence = []
        self.delay, self.area = float('inf'), float('inf')

        self.best_known_area = (float('inf'), float('inf'), -1, -1)
        self.best_known_delay = (float('inf'), float('inf'), -1, -1)
        self.best_known = (float('inf'), float('inf'), -1, -1)
        self.episode_best_area_meets_constraint = float('inf')

        self.ref_delay = ref_delay
        self.ref_area = ref_area
        self.metric = metric

        self.best_score = self.score(delay=self.delay, area=self.delay)

        # logging
        self.log = None

    def score(self, area, delay):
        """ Score to maximize """
        if self.metric == 'sum':
            return (1 - (area / self.ref_area)) + (1 - (delay / self.ref_delay))
        elif self.metric == 'min':
            return min((1 - (area / self.ref_area)), (1 - (delay / self.ref_delay)))
        elif self.metric == 'constraint':
            if delay > self.delay_constr:
                return (1 - (delay / self.ref_delay)) - 10
            else:
                return 1 - (area / self.ref_area)
        raise ValueError(self.metric)

    @property
    def design(self) -> str:
        """ Design name """
        return os.path.basename(self.design_file).split('.')[0]

    def __del__(self):
        if self.log:
            self.log.close()

    def constr_met(self) -> bool:
        """ Whether constraint is met """
        return self.delay <= self.delay_constr

    def reset(self):
        """
        resets the environment and returns the state
        """
        self.iteration = 0
        self.episode += 1
        self.current_design_file = self.design_file

        self.delay, self.area = float('inf'), float('inf')
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.playground_dir, str(self.episode))
        self.episode_best_area_meets_constraint = float('inf')
        os.makedirs(self.episode_dir, exist_ok=True)

        # logging
        log_file = os.path.join(self.episode_dir, 'log.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, area, delay, best_area_meets_constraint, best_area, best_delay\n')

        # logging
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(self.area), str(self.delay)]) + '\n')
        self.log.flush()

    @property
    def library_file(self) -> str:
        return self.params['mapping']['library_file']

    @property
    def abc_binary(self) -> str:
        return self.params['abc_binary']

    @property
    def delay_constr(self) -> float:
        return self.params['mapping']['clock_period']

    @property
    def post_mapping_optimizations(self) -> float:
        return self.params['post_mapping_commands']

    @property
    def optimizations(self) -> List[str]:
        return self.params['optimizations']

    def run_episode(self, k: int):
        for i in range(self.max_iterations):
            self.iteration += 1
            # log
            log('Iteration: ' + str(i + 1))
            log('-------------' * 3)

            # create a directory for this iteration
            iteration_dir = os.path.join(self.episode_dir, str(i))
            os.makedirs(iteration_dir, exist_ok=True)

            optimizations = np.random.choice(self.optimizations, size=k, replace=False)
            # in parallel, run ABC on each of the optimizations we have
            results = Parallel(n_jobs=len(optimizations))(
                delayed(run_thread)(iteration_dir, self.current_design_file, opt, self.library_file, self.delay_constr)
                for opt in optimizations)

            # get the minimum result of all threads
            best_thread = max(results, key=lambda t: self.score(delay=t[2], area=t[3]))  # getting best action

            # hold the best result in variables
            best_optimization = best_thread[0]
            best_optimization_file = best_thread[1]
            self.delay = best_thread[2]
            self.area = best_thread[3]
            score = self.score(delay=self.delay, area=self.area)

            if self.area < self.best_known_area[0]:
                self.best_known_area = (self.area, self.delay, self.episode, self.iteration)
            if self.delay < self.best_known_delay[1]:
                self.best_known_delay = (self.area, self.delay, self.episode, self.iteration)
            if score > self.best_score:
                self.best_known = (self.area, self.delay, self.episode, self.iteration)
                self.best_score = score
            if score > -float('inf') and self.area < self.episode_best_area_meets_constraint:
                self.episode_best_area_meets_constraint = self.area

            log(f'Choosing Optimization for {self.design} ({k}): ' + best_optimization + ' -> delay: ' + str(self.delay) + ', area: ' + str(
                self.area) + f' | score: {score}')
            self.log.write(', '.join(list(map(str, [self.iteration, best_optimization, self.area, self.delay]))) + ', ' +
                           '; '.join(list(map(str, self.best_known))) + ', ' +
                           '; '.join(list(map(str, self.best_known_area))) + ', ' +
                           '; '.join(list(map(str, self.best_known_delay))) + '\n')
            self.log.flush()
            # update design file for the next iteration
            self.current_design_file = best_optimization_file
            log('================')
        n_designs_to_remove = 0
        for design in glob.glob(self.episode_dir + '/*/*/*.blif'):
            os.remove(design)
            n_designs_to_remove += 1
        log(f'Remove {n_designs_to_remove} design files in {self.episode_dir}')
