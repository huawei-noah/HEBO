import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from utilities.constraint_utils import check_constraint_satisfaction_batch
from utilities.utils import AA_to_idx, sample_to_aa_seq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RandomSearch:

    def __init__(self, function, dimension, num_iter, batch_size, save_dir, convergence_curve=True):

        #############################################################
        # input function
        assert (callable(function)), "function must be callable"
        self.f = function

        #############################################################
        # dimension
        self.dim = int(dimension)

        #############################################################
        # input variables' boundaries
        self.var_bound = np.array([0, len(AA_to_idx) - 1])

        #############################################################
        # convergence_curve
        if convergence_curve == True:
            self.convergence_curve = True
        else:
            self.convergence_curve = False

        #############################################################
        assert batch_size <= num_iter

        # Number of random samples
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.num_iter / self.batch_size))

        #############################################################
        # Directory to store results
        self.save_dir = save_dir
        # try:
        #     os.mkdir(self.save_dir)
        # except:
        #     raise Exception(
        #         "Save directory already exists. Choose a different directory to avoid overwriting previous results.")

    def run(self):

        #############################################################
        # Dataframe used to store all results
        self.res = pd.DataFrame(np.nan, index=np.arange(int(self.num_iter) + 1), columns=['Index', 'LastValue',
                                                                                          'BestValue', 'Time',
                                                                                          'LastProtein',
                                                                                          'BestProtein'])

        #############################################################
        # Variable to keep track of total number of function evaluations
        self.num_funct_evals = 0
        self.best_sequence = None
        self.best_function = None

        for batch in range(self.num_batches):
            #############################################################
            # Generate the required number of samples
            start = time.time()
            if batch < self.num_batches - 1:
                pop_size = self.batch_size
            else:
                pop_size = self.num_iter - (self.num_batches - 1) * self.batch_size

            # Create the initial population. Last column stores the fitness
            population = np.zeros(shape=(pop_size, self.dim + 1))
            population[:, :self.dim] = np.random.randint(low=self.var_bound[0], high=self.var_bound[1] + 1,
                                                         size=(pop_size, self.dim))

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

            # Save all the data
            self.save_results()
            np.save(os.path.join(self.save_dir, f'population_batch_{batch + 1}_out_of_{self.num_batches}.npy'),
                    population)

        #############################################################
        # Print and save all results

        sys.stdout.write('\r The best solution found:\n %s' % (self.res.iloc[-1]['BestProtein']))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.res.iloc[-1]['BestValue']))
        sys.stdout.flush()

        return self.res

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
            aa_seq = sample_to_aa_seq(temp[X_idx])
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

    def save_results(self):

        self.res.to_csv(os.path.join(self.save_dir, 'results.csv'))

        if self.convergence_curve == True:
            plt.figure()
            plt.title("Random Search Binding energy curve")
            plt.grid()
            plt.plot(self.res.iloc[1:self.num_funct_evals + 1]['Index'].astype(int),
                     self.res.iloc[1:self.num_funct_evals + 1]['BestValue'])
            plt.xlabel('Number of function evaluations')
            plt.ylabel('Minimum Binding Energy')
            plt.savefig(os.path.join(self.save_dir, "binding_energy_vs_funct_evals.png"))
            plt.close()

        return
