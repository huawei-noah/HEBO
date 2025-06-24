import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from task.tools import Absolut
from utilities.config_utils import load_config
from genetic_algorithm.actor import GeneticAlgorithmActor
from environment.binding_environment import BindingEnvironment
import pandas as pd
from utilities.config_utils import save_config


def summarisation(res, save_dir, num_funct_evals, convergence_curve=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    res.to_csv(os.path.join(save_dir, 'results.csv'))

    if convergence_curve == True:
        plt.figure()
        plt.title("Genetic Algorithm Binding energy curve")
        plt.grid()
        plt.plot(res.iloc[1:num_funct_evals + 1]['Index'].astype(int),
                 res.iloc[1:num_funct_evals + 1]['BestValue'])
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Minimum Binding Energy')
        plt.savefig(os.path.join(save_dir, "binding_energy_vs_funct_evals.png"))
        plt.close()

    return


def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush()


def calculate_total_num_funct_evals(algorithm_parameters):
    population_size = algorithm_parameters['population_size']
    num_iterations = algorithm_parameters['max_num_iterations']
    num_elite = int(population_size * algorithm_parameters['elite_ratio'])
    if num_elite % 2 != 0:  # Ensure that the number of elite samples is even
        num_elite += 1

    return population_size + num_iterations * (population_size - num_elite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Script to visualise the CDR3-antigen binding '
                                                 'for various methods at various timesteps')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration File')
    args = parser.parse_args()
    config = load_config(args.config)

    # Main Parameters

    absolut_config = {"antigen": None,
                      "path": config['absolut_config']['path'],
                      "process": config['absolut_config']['process'],
                      'startTask': config['absolut_config']['startTask']}

    # GA parameters
    algorithm_parameters = {'max_num_iterations': config['genetic_algorithm_config']['max_num_iterations'],
                            'population_size': config['genetic_algorithm_config']['population_size'],
                            'mutation_probability': 1 / config['sequence_length'],
                            'elite_ratio': config['genetic_algorithm_config']['elite_ratio'],
                            'crossover_probability': config['genetic_algorithm_config']['crossover_probability'],
                            'parents_portion': config['genetic_algorithm_config']['parents_portion'],
                            'crossover_type': config['genetic_algorithm_config']['crossover_type']}

    # Print the number of black-box function evaluations per antigen per random seed
    print(
        f"\nNumber of function evaluations per random seed: {calculate_total_num_funct_evals(algorithm_parameters)}")

    # Create a directory to save all results
    """
    try:
        os.mkdir(config['save_dir'])
    except:
        raise Exception("Save directory already exists. Choose another directory name to avoid overwriting data")
    """

    with open('dataloader/all_antigens.txt') as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]
    print(f"Running over all input antigens from file: \n \n {antigens} \n")
    for antigen in tqdm(antigens):
        absolut_config['antigen'] = antigen

        # Defining the fitness function
        absolut_binding_energy = Absolut(absolut_config)


        def function(x):
            x = x.astype(int)
            return absolut_binding_energy.energy(x)


        environment_config = {'n_sequences': config['genetic_algorithm_config']['population_size'],
                              'dimensions': config['sequence_length'],
                              'model_tag': antigen}

        env = BindingEnvironment(env_name="one_shot", config=environment_config, evaluator=function)

        binding_energy = []
        num_function_evals = []

        print(f"\nAntigen: {antigen}")

        for i, seed in enumerate(config['random_seeds']):
            start = time.time()
            print(f"\nRandom seed {i + 1}/{len(config['random_seeds'])}")

            np.random.seed(seed)
            _save_dir = os.path.join(
                config['save_dir'],
                f"antigen_{antigen}_kernel_GA_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")

            if os.path.exists(_save_dir):
                print(f"antigen_{antigen}_kernel_GA_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")
                continue

            actor = GeneticAlgorithmActor(config)
            max_num_funct_evals = calculate_total_num_funct_evals(algorithm_parameters)

            # results = ResultsLogger(max_num_funct_evals)  or  initialise instance outside of loop and just call
            # results.reset() here

            res = pd.DataFrame(np.nan, index=np.arange(int(max_num_funct_evals) + 1),
                               columns=['Index', 'LastValue',
                                        'BestValue', 'Time',
                                        'LastProtein',
                                        'BestProtein'])

            dim = 11

            max_num_iter = int(algorithm_parameters['max_num_iterations'])

            #############################################################
            # Variable to keep track of total number of function evaluations
            num_funct_evals = 0
            best_sequence = None
            best_function = None

            #############################################################
            # Initial Population. Last column stores the fitness
            start_generation = time.time()
            population, population_to_eval = actor.suggest()
            end_generation = time.time()

            _, fitness, _, _ = env.step(population_to_eval)

            res, population, nf = actor.observe(population_to_eval, fitness, end_generation - start_generation,
                                                num_funct_evals)
            # results.append_batch()
            num_funct_evals += len(population_to_eval)

            ##############################################################
            gen_num = 1

            while gen_num <= max_num_iter:
                progress(gen_num, max_num_iter, status="GA is running...")

                summarisation(res, _save_dir, nf)

                start_generation = time.time()
                population, population_to_eval = actor.suggest()
                end_generation = time.time()

                _, fitness, _, _ = env.step(population_to_eval)

                res, population, nf = actor.observe(population, fitness, end_generation - start_generation,
                                                    num_funct_evals)
                # results.append_batch()
                num_funct_evals += len(population_to_eval)

                gen_num += 1

            sys.stdout.write('\r The best solution found:\n %s' % (res.iloc[-1]['BestProtein']))
            sys.stdout.write('\n\n Objective function:\n %s\n' % (res.iloc[-1]['BestValue']))
            sys.stdout.flush()

            # results.save(_save_dir)
            summarisation(res, _save_dir, num_funct_evals)
            save_config(algorithm_parameters, os.path.join(_save_dir, '../config.yaml'))
            np.save(os.path.join(_save_dir, 'final_population.npy'), population)

            binding_energy.append(res['BestValue'].to_numpy())
            num_function_evals.append(res['Index'].to_numpy())
            print("\nTime taken to run Genetic algorithm for a single seed: {:.0f}s".format(time.time() - start))

        binding_energy = np.array(binding_energy)
        num_function_evals = np.array(num_function_evals)

        np.save(os.path.join(config['save_dir'], f"binding_energy_{antigen}.npy"), binding_energy)
        np.save(os.path.join(config['save_dir'], f"num_function_evals_{antigen}.npy"), num_function_evals)

        n_std = 1

        plt.figure()
        plt.title(f"Genetic Algorithm {antigen}")
        plt.grid()
        plt.plot(num_function_evals[0], np.mean(binding_energy, axis=0), color="b")
        plt.fill_between(num_function_evals[0],
                         np.mean(binding_energy, axis=0) - n_std * np.std(binding_energy, axis=0),
                         np.mean(binding_energy, axis=0) + n_std * np.std(binding_energy, axis=0),
                         alpha=0.2, color="b")
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Minimum Binding Energy')
        plt.savefig(os.path.join(config['save_dir'], f"binding_energy_vs_funct_evals_{antigen}.png"))
        plt.close()
