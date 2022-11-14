import os
import sys
import time

import numpy as np

from pathlib import Path

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
# sys.path.insert(0, ROOT_PROJECT)

from task.tools import Absolut
from utilities.config_utils import load_config
from genetic_algorithm.genetic_algorithms import calculate_total_num_funct_evals, GeneticAlgorithm
from actor.base import BaseActor
from utilities.constraint_utils import check_constraint_satisfaction_batch
from utilities.aa_utils import aa_to_idx, indices_to_aa_seq
import pandas as pd


def calculate_total_num_funct_evals(algorithm_parameters):
    population_size = algorithm_parameters['population_size']
    num_iterations = algorithm_parameters['max_num_iterations']
    num_elite = int(population_size * algorithm_parameters['elite_ratio'])
    if num_elite % 2 != 0:  # Ensure that the number of elite samples is even
        num_elite += 1

    return population_size + num_iterations * (population_size - num_elite)

class GeneticAlgorithmActor(BaseActor):

    def __init__(self, config):
        self.absolut_config = {"antigen": None,
                          "path": config['absolut_config']['path'],
                          "process": config['absolut_config']['process'],
                          'startTask': config['absolut_config']['startTask']}

        # GA parameters
        algorithm_parameters = {'max_num_iterations': config['genetic_algorithm_config']['max_num_iterations'], \
                                'population_size': config['genetic_algorithm_config']['population_size'], \
                                'mutation_probability': 1 / config['sequence_length'], \
                                'elite_ratio': config['genetic_algorithm_config']['elite_ratio'], \
                                'crossover_probability': config['genetic_algorithm_config']['crossover_probability'], \
                                'parents_portion': config['genetic_algorithm_config']['parents_portion'], \
                                'crossover_type': config['genetic_algorithm_config']['crossover_type']}

        self.iter = 0
        self.dim = int(config['sequence_length'])
        self.var_bound = np.array([0, len(aa_to_idx) - 1])
        self.population_size = int(algorithm_parameters['population_size'])
        self.sorted_population = None
        self.replay_buffer = []
        self.best_sequence = None
        self.best_function = None
        max_num_funct_evals = calculate_total_num_funct_evals(algorithm_parameters)
        self.num_parents = int(algorithm_parameters['parents_portion'] * self.population_size)
        
        self.res = pd.DataFrame(np.nan, index=np.arange(int(max_num_funct_evals) + 1), columns=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])

        trl = self.population_size * algorithm_parameters['elite_ratio']
        if trl < 1 and algorithm_parameters['elite_ratio'] > 0:
            self.num_elite = 1
        else:
            self.num_elite = int(trl)

        if self.num_elite % 2 != 0:  # Ensure that the number of elite samples is even
            self.num_elite += 1

        assert (self.num_parents >= self.num_elite), "\n number of parents must be greater than number of elite samples"

        # Maximum number of iterations
        assert (algorithm_parameters['max_num_iterations'] > 0), "\n maximum number of iterations must be grater than 0"
        self.max_num_iter = int(algorithm_parameters['max_num_iterations'])

        # Crossover type
        self.crossover_type = algorithm_parameters['crossover_type']
        assert (self.crossover_type in ['uniform', 'one_point', 'two_point']), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"
            
                # Mutation probability
        self.mutation_prob = algorithm_parameters['mutation_probability']
        assert (0 <= self.mutation_prob <= 1), "mutation_probability must be in range [0,1]"

        # Crossover probability
        self.crossover_prob = algorithm_parameters['crossover_probability']
        assert (0 <= self.crossover_prob <= 1), "mutation_probability must be in range [0,1]"

    def sample_initial_population(self):
        # Generate initial population using rejection sampling to ensure constraint satisfaction

        start = time.time()
        # Create the initial population. Last column stores the fitness
        population = np.zeros(shape=(self.population_size, self.dim + 1))
        population[:, :self.dim] = np.random.randint(low=self.var_bound[0], high=self.var_bound[1] + 1,
                                                     size=(self.population_size, self.dim))

        # Check for constraint violation
        constraints_violated = np.logical_not(check_constraint_satisfaction_batch(population[:, :self.dim]))

        # Continue until all samples satisfy the constraints
        while np.sum(constraints_violated) != 0:
            # Generate new samples for the ones that violate the constraints
            population[constraints_violated, :self.dim] = np.random.randint(
                low=self.var_bound[0], high=self.var_bound[1] + 1, size=(np.sum(constraints_violated), self.dim))

            # Check for constraint violation
            constraints_violated = np.logical_not(check_constraint_satisfaction_batch(population[:, :self.dim]))

        time_to_generate_population = time.time() - start
        # Get the fitness of the initial population
        #population[:, self.dim] = self.evaluate_batch(population[:, :self.dim], time_to_generate_population)
        return population

    def sample_new_population(self, sorted_population):

        ##############################################################
        # Normalizing objective function
        minobj = sorted_population[0, self.dim]
        if minobj < 0:
            normobj = sorted_population[:, self.dim] + abs(minobj)

        else:
            normobj = sorted_population[:, self.dim].copy()

        maxnorm = np.amax(normobj)
        normobj = maxnorm - normobj + 1

        #############################################################
        # Calculate probability

        sum_normobj = np.sum(normobj)
        prob = normobj / sum_normobj
        cumprob = np.cumsum(prob)

        #############################################################
        # Select parents
        parents = np.zeros(shape=(self.num_parents, self.dim + 1))

        # First, append the best performing samples to the list of parents
        for k in range(0, self.num_elite):
            parents[k] = sorted_population[k].copy()

        # Then append random samples to the list of parents. The probability of a sample being picked is
        # proportional to the fitness of a sample
        for k in range(self.num_elite, self.num_parents):
            index = np.searchsorted(cumprob, np.random.random())
            parents[k] = sorted_population[index].copy()

        #############################################################
        # New generation
        new_population = np.zeros(shape=(self.population_size, self.dim + 1))

        # First, all Elite samples from the previous population are added to the new population
        for k in range(0, self.num_elite):
            new_population[k] = parents[k].copy()

        # Second, perform crossover with the previously determined subset of all the parents. Do not evaluate
        # the new samples yet to increase efficiency
        for k in range(self.num_elite, self.population_size, 2):
            r1 = np.random.randint(0, self.num_parents)
            r2 = np.random.randint(0, self.num_parents)
            pvar1 = parents[r1, : self.dim].copy()
            pvar2 = parents[r2, : self.dim].copy()

            # Constraint satisfaction with rejection sampling
            constraints_satisfied = False
            while not constraints_satisfied:
                ch = self.crossover(pvar1, pvar2, self.crossover_type)
                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                ch1 = self.mut(ch1)
                ch2 = self.mut(ch2)

                constraints_satisfied = check_constraint_satisfaction_batch(np.array([ch1, ch2])).all()

            new_population[k, :self.dim] = ch1.copy()
            new_population[k + 1, :self.dim] = ch2.copy()

        return new_population

        ##############################################################################

    def crossover(self, x, y, c_type):

        # children are copies of parents by default
        ofs1, ofs2 = x.copy(), y.copy()

        # Do not perform crossover on all offsprings
        if np.random.random() <= self.crossover_prob:

            if c_type == 'one_point':
                ran = np.random.randint(0, self.dim)
                for i in range(0, ran):
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

            if c_type == 'two_point':

                ran1 = np.random.randint(0, self.dim)
                ran2 = np.random.randint(ran1, self.dim)

                for i in range(ran1, ran2):
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

            if c_type == 'uniform':

                for i in range(0, self.dim):
                    ran = np.random.random()
                    if ran < 0.5:
                        ofs1[i] = y[i].copy()
                        ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])

        ###############################################################################

    def mut(self, x):

        for i in range(self.dim):
            ran = np.random.random()
            if ran < self.mutation_prob:
                x[i] = np.random.randint(self.var_bound[0], self.var_bound[1] + 1)

        return x

    ###############################################################################

    def evaluate_batch(self, X, fitness, time_to_generate_population=0, num_funct_evals=0):
        #############################################################
        # Function to evaluate a batch of samples
        start = time.time()
        temp = X.copy()
        batch_size = len(X)
        time_to_evaluate_population = time.time() - start

        mean_time_per_sample = (time_to_generate_population + time_to_evaluate_population) / batch_size

        for X_idx, res_idx in enumerate(range(num_funct_evals, num_funct_evals + batch_size)):
            aa_seq = indices_to_aa_seq(temp[X_idx])
            binding_energy = fitness[X_idx]

            # Initialisation during first function evaluation
            if (self.best_function is None and self.best_sequence is None):
                self.best_function = binding_energy
                self.best_sequence = aa_seq

            # Check if binding energy of current sequence is lower than the binding energy of the best sequence
            elif (binding_energy < self.best_function):
                self.best_function = binding_energy
                self.best_sequence = aa_seq

            # Append results 'Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'

            self.res.iloc[res_idx + 1] = {'Index': int(res_idx) + 1, 'LastValue': binding_energy,
                                          'BestValue': self.best_function, 'Time': mean_time_per_sample,
                                          'LastProtein': aa_seq, 'BestProtein': self.best_sequence, }

        return self.res

    def suggest(self):
        if(self.iter == 0):
            population = self.sample_initial_population()
        else:
            population = self.sample_new_population(self.sorted_population)

        self.iter += 1

        return population

    def observe(self, population, fitness, generation_time, num_funct_evals):
        res = self.evaluate_batch(population, fitness, generation_time, num_funct_evals)
        print(population.shape)
        self.sorted_population = population[population[:, self.dim].argsort()]
        return res, self.sorted_population

