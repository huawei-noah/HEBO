import os
import sys
import time
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

import numpy as np
import pandas as pd

from utilities.config_utils import save_config
from utilities.aa_utils import aa_to_idx, indices_to_aa_seq
from utilities.constraint_utils import check_constraint_satisfaction_batch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Calculate total number of function evaluations
def calculate_total_num_funct_evals(algorithm_parameters):
    population_size = algorithm_parameters['population_size']
    num_iterations = algorithm_parameters['max_num_iterations']
    num_elite = int(population_size * algorithm_parameters['elite_ratio'])
    if num_elite % 2 != 0:  # Ensure that the number of elite samples is even
        num_elite += 1

    return population_size + num_iterations * (population_size - num_elite)


class GeneticAlgorithm:
    ''' Implementation based on https://github.com/rmsolgi/geneticalgorithm

    Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations
    '''

    ##############################################################################

    def __init__(self, function,
                 dimension,
                 save_dir,
                 algorithm_parameters={'max_num_iterations': 3, \
                                       'population_size': 10, \
                                       'mutation_probability': 0.1, \
                                       'elite_ratio': 0.1, \
                                       'crossover_probability': 0.5, \
                                       'parents_portion': 0.3, \
                                       'crossover_type': 'uniform'}, \
                 convergence_curve=True, \
                 progress_bar=True):

        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param save_path <string> - path to directory where all data will be saved

        @param dimension <integer> - the number of decision variables

        @param algorithm_parameters:
            @ max_num_iterations <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elite_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iterations_without_improvement <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''

        #############################################################
        # input function
        assert (callable(function)), "function must be callable"
        self.f = function

        #############################################################
        # dimension
        self.dim = int(dimension)

        #############################################################
        # input variables' boundaries
        self.var_bound = np.array([0, len(aa_to_idx) - 1])

        #############################################################
        # convergence_curve
        if convergence_curve == True:
            self.convergence_curve = True
        else:
            self.convergence_curve = False

        #############################################################
        # progress_bar
        if progress_bar == True:
            self.progress_bar = True
        else:
            self.progress_bar = False

        #############################################################
        # Directory to store results
        self.save_dir = save_dir

        #############################################################

        #############################################################
        # input algorithm's parameters
        self.param = algorithm_parameters

        # Size of entire population
        self.population_size = int(self.param['population_size'])

        # Number of parents
        assert (0 <= self.param['parents_portion'] <= 1), "parents_portion must be in range [0,1]"
        self.num_parents = int(self.param['parents_portion'] * self.population_size)

        # Mutation probability
        self.mutation_prob = self.param['mutation_probability']
        assert (0 <= self.mutation_prob <= 1), "mutation_probability must be in range [0,1]"

        # Crossover probability
        self.crossover_prob = self.param['crossover_probability']
        assert (0 <= self.crossover_prob <= 1), "mutation_probability must be in range [0,1]"

        # Number of elite samples per iteration
        assert (0 <= self.param['elite_ratio'] <= 1), "elite_ratio must be in range [0,1]"
        trl = self.population_size * self.param['elite_ratio']
        if trl < 1 and self.param['elite_ratio'] > 0:
            self.num_elite = 1
        else:
            self.num_elite = int(trl)

        if self.num_elite % 2 != 0:  # Ensure that the number of elite samples is even
            self.num_elite += 1

        assert (self.num_parents >= self.num_elite), "\n number of parents must be greater than number of elite samples"

        # Maximum number of iterations
        assert (self.param['max_num_iterations'] > 0), "\n maximum number of iterations must be grater than 0"
        self.max_num_iter = int(self.param['max_num_iterations'])

        # Crossover type
        self.crossover_type = self.param['crossover_type']
        assert (self.crossover_type in ['uniform', 'one_point', 'two_point']), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

    ##############################################################################

    def run(self):

        max_num_funct_evals = calculate_total_num_funct_evals(self.param)

        #############################################################
        # Dataframe used to store all results
        self.res = pd.DataFrame(np.nan, index=np.arange(int(max_num_funct_evals) + 1), columns=['Index', 'LastValue',
                                                                                            'BestValue', 'Time',
                                                                                            'LastProtein',
                                                                                            'BestProtein'])

        #############################################################
        # Variable to keep track of total number of function evaluations
        self.num_funct_evals = 0
        self.best_sequence = None
        self.best_function = None

        #############################################################
        # Initial Population. Last column stores the fitness
        population = self.sample_initial_population()

        #############################################################
        # Sort
        sorted_population = population[population[:, self.dim].argsort()]

        ##############################################################
        gen_num = 1
        while gen_num <= self.max_num_iter:

            if self.progress_bar == True:
                self.progress(gen_num, self.max_num_iter, status="GA is running...")

            # Save all current results
            _ = self.save_results()

            population = self.sample_new_population(sorted_population)

            #############################################################
            # Sort
            sorted_population = population[population[:, self.dim].argsort()]

            #############################################################
            gen_num += 1

        #############################################################

        if self.progress_bar == True:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))

        sys.stdout.write('\r The best solution found:\n %s' % (self.res.iloc[-1]['BestProtein']))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.res.iloc[-1]['BestValue']))
        sys.stdout.flush()

        # Save all the data
        self.save_results()
        save_config(self.param, os.path.join(self.save_dir, 'config.yaml'))
        np.save(os.path.join(self.save_dir, 'final_population.npy'), sorted_population)

        return self.res

    ##############################################################################

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
        population[:, self.dim] = self.evaluate_batch(population[:, :self.dim], time_to_generate_population)

        return population

    ##############################################################################

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

        # Evaluate all new samples
        outs = self.evaluate_batch(new_population[self.num_elite:, :self.dim])
        n_elites = outs.shape[0]
        new_population[self.num_elite:(self.num_elite+n_elites), self.dim] = outs[:n_elites]
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

    def evaluate_batch(self, X, time_to_generate_population=0):
        #############################################################
        # Function to evaluate a batch of samples
        start = time.time()
        temp = X.copy()
        batch_size = len(X)
        fitness, _ = self.f(temp)
        time_to_evaluate_population = time.time() - start

        mean_time_per_sample = (time_to_generate_population + time_to_evaluate_population) / batch_size

        for X_idx, res_idx in enumerate(range(self.num_funct_evals, self.num_funct_evals + batch_size)):
            aa_seq = indices_to_aa_seq(temp[X_idx])
            try:
                binding_energy = fitness[X_idx]
            except IndexError:
                binding_energy = 0.0

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

        # Increment the number of function evaluations
        self.num_funct_evals += batch_size

        return fitness

    ###############################################################################

    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()

    ###############################################################################

    def save_results(self):

        self.res.to_csv(os.path.join(self.save_dir, 'results.csv'))

        if self.convergence_curve == True:
            plt.figure()
            plt.title("Genetic Algorithm Binding energy curve")
            plt.grid()
            plt.plot(self.res.iloc[1:self.num_funct_evals + 1]['Index'].astype(int),
                     self.res.iloc[1:self.num_funct_evals + 1]['BestValue'])
            plt.xlabel('Number of function evaluations')
            plt.ylabel('Minimum Binding Energy')
            plt.savefig(os.path.join(self.save_dir, "binding_energy_vs_funct_evals.png"))
            plt.close()

        return
