import time
import logging
import numpy as np
import networkx as nx
import GPyOpt
from GPyOpt.util.general import normalize
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from myAcquisitionModular import MyAcquisitionModular
from myGPModel import MyGPModel
from acquisition_optimizer import MPAcquisitionOptimizer, BruteForceAcquisitionOptimizer
from GPyOpt.core.evaluators.sequential import Sequential
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.core.task.cost import CostModel

import networkx as nx

class BOStopper(GPyOpt.core.BO):
    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1, de_duplication = False):

        super(BOStopper, self).__init__(model, space, objective, acquisition, evaluator, X_init, Y_init, cost, normalize_Y, model_update_interval, de_duplication)

    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None, stopping_Y=None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)
        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)
                    or np.any(self.Y_new == stopping_Y)):
                break

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)


class MyBOModular(BOStopper):
    
    def __init__(self, domain, initial_design, graph_function,
        normalize_Y=False, max_eval=-1,
        fn=None, fn_optimizer=None, noise_var=None, exact_feval=None, exploration_weight_function=None, learnDependencyStructureRate=None, learnParameterRate=None,
        acq_opt_restarts=1, random_graph=False, additional_args=dict()):

        #self.design_space = Design_space(domain.get_gpy_domain())
        self.t = 0
        self.fn = fn
        self.objective = SingleObjective(self.fn, 1, "no name", space=domain)
        self._init_design(initial_design)
        self.random_graph = random_graph
        self.additional_args = additional_args

        self.domain = domain

        acquisition_optimizer_type = additional_args.get('acquisition_optimizer', 'MP')
        ensemble_samples = additional_args.get('ensemble_samples', 1)
        size_of_random_graph = additional_args.get('size_of_random_graph', None)
        
        # model needed for LCB
        self.model = MyGPModel(noise_var=noise_var, exact_feval=exact_feval, optimize_restarts=0, 
                exploration_weight_function=exploration_weight_function, learnDependencyStructureRate=learnDependencyStructureRate, 
                learnParameterRate=learnParameterRate, 
                graph_function=graph_function.copy_randomised(size_of_random_graph) if random_graph else graph_function.copy(), 
                mlflow_logging=self.fn.mlflow_logging, 
                fn=self.fn, random_graph=random_graph, additional_args=additional_args)

        if acquisition_optimizer_type == 'MP':
            self.acquisition_optimizer = MPAcquisitionOptimizer(domain, self.model.graph_function, [], self.fn.mlflow_logging, max_eval=max_eval, acq_opt_restarts=acq_opt_restarts)
        elif acquisition_optimizer_type == 'bf':
            self.acquisition_optimizer = BruteForceAcquisitionOptimizer(domain, [], self.fn.mlflow_logging, max_eval=max_eval, acq_opt_restarts=acq_opt_restarts)
        elif acquisition_optimizer_type in {'lbfgs', "DIRECT"}:
            self.acquisition_optimizer = AcquisitionOptimizer(domain, optimizer=acquisition_optimizer_type)

        ## !!! models inside acqu1 must be the same as models in MyModel !!! -> Ok in Python, the object are references, not copied
        self.acquisition = MyAcquisitionModular(self.model, self.acquisition_optimizer, domain)
        self.evaluator = Sequential(self.acquisition)
        
        self.modular_optimization = False
        
        self.cost = CostModel(None)
        self.fn_optimizer = fn_optimizer

        super(MyBOModular, self).__init__(model = self.model, space = domain, objective = self.objective,
            acquisition = self.acquisition, evaluator = self.evaluator, X_init = self.X, Y_init = self.Y,
            cost = self.cost, normalize_Y = normalize_Y, model_update_interval = 1)

    def _init_design(self, initial_design):
        # in case function provides a fixed initial design
        if hasattr(self.fn, 'get_default_values'):
            self.X, self.Y = self.fn.get_default_values(), self.objective.evaluate(self.fn.get_default_values())[0]
            
            if len(initial_design) > 0:
                self.X = np.concatenate([self.X, initial_design])
                Y, _ = self.objective.evaluate(initial_design)
                self.Y = np.concatenate([self.Y, Y])

        else:
            self.X = initial_design
            self.Y, _ = self.objective.evaluate(self.X)

    def _update_model(self, normalization_type):
        """
        Updates the model and saves the parameters (if available).
        """
        if self.t == 0:
            # Attempt to return f instead of y if that exist
            self.fn.mlflow_logging.log_init_y(np.min(self.Y))
        else:
            self.fn.mlflow_logging.log_y(np.min(self.Y[-1]))
        self.t += 1
        self.model.update_structure(self.acquisition, self.X, self.Y, self.fn_optimizer, self.fn)
        if self.num_acquisitions % self.model_update_interval == 0:

            # input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)

            # Y_inmodel is the output that goes into the model
            if self.normalize_Y:
                Y_inmodel = normalize(self.Y, normalization_type)
            else:
                Y_inmodel = self.Y

            self.model.updateModel(X_inmodel, Y_inmodel, None, None, log=False)

        # Save parameters of the model
        self._save_model_parameter_values()
    
    def _save_model_parameter_values(self):
        return
