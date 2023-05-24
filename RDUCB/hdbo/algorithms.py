import logging

import mlflow

import numpy as np
import GPyOpt
import GPy
#from numpy.random import seed
from GPyOpt.models.gpmodel import GPModel
from GPyOpt.acquisitions import AcquisitionLCB
import networkx as nx
import collections
from myBOModular import MyBOModular
from myGPModel import MyGPModel
from GPyOpt.core.task.space import Design_space
from common import Config
import random
import os
import pickle

# CLEAN UP?
from function_optimizer import GraphOverlap, GraphNonOverlap, Tree, GraphFunction, OptimalGraphFunction
from graph_utils import get_random_graph

from exceptions import EarlyTerminationException

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

from datasets import ComponentFunction, SyntheticComponentFunction
import function_optimizer

class MetaLoader(type):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, MetaLoader).__new__(cls, cls_name, bases, attrs)
        MetaLoader.registry[cls_name] = new_class
        MetaLoader.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(loader_id):
        logging.info("Load algorithm loader[%s].", loader_id)
        return MetaLoader.registry[loader_id]

class Algorithm(type, metaclass=MetaLoader):
    registry = {}
    algorithm_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Algorithm).__new__(cls, cls_name, bases, attrs)
        Algorithm.registry[cls_name] = new_class
        Algorithm.algorithm_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_constructor(algorithm_id):
        logging.info("Using algorithm with algorithm_id[%s].", algorithm_id)
        return Algorithm.registry[algorithm_id]

from febo.models.gp import GPConfig
from febo.controller.simple import SimpleControllerConfig
from febo.environment.benchmarks import BenchmarkEnvironmentConfig
from febo.solvers.candidate import GridSolverConfig
from febo.algorithms.rembo import RemboConfig
from febo.models.model import ModelConfig

from febo.environment.benchmarks import BenchmarkEnvironment
from febo.environment import DiscreteDomain, ContinuousDomain
from febo.controller import SimpleController
import febo

class AdaptorBenchmark(BenchmarkEnvironment):
    def __init__(self, fn):
        super().__init__(path=None)
        self.fn = fn
        self.mlflow_logging = self.fn.mlflow_logging
        dim = self.fn.domain.dimension
        L = []
        U = []
        # Number of points per dimension
        n_points = []
        # Go through each domain of the dimension and find the l and u
        for d in self.fn.domain.combined_domain:
            L.append(np.min(d))
            U.append(np.max(d))
            n_points.append(len(d))
            
        self._domain = ContinuousDomain(np.array(L), np.array(U))
        #GridSolverConfig.points_per_dimension = np.max(n_points)
        
        RemboConfig.emb_d = self.fn.get_emb_dim()
        
        # ??
        #self._domain = DiscreteDomain(np.array([[0.0, 0.0], [1.0, 1.0]]))
        
        self._max_value = -self.mlflow_logging.y_opt
    def f(self, x):
        return np.float64(-self.fn(np.array([x])))

class GADDUCBAlgorithm(object):
    def __init__(self, n_iter, algorithm_random_seed, n_rand, algoID="", fn=None, **kwargs):
        self.algoID = algoID
        self.n_iter = n_iter
        self.domain = fn.domain
        self.fn = fn
        self.algorithm_random_seed = algorithm_random_seed
        self.n_rand = n_rand
        # Use the same Random Seed everywhere
        # generate init design depends on the random seed setting.
        np.random.seed(algorithm_random_seed)
        random.seed(algorithm_random_seed)
        self.rs = np.random.RandomState(algorithm_random_seed)
        self.initial_design = self.domain.random_X(self.rs, n_rand)
    def get_algorithm_id(self):
        return self.__class__.__name__ + self.algoID
    def run(self):
        raise NotImplementedError


class FEBOAlgorithm(GADDUCBAlgorithm):
    def __init__(self, initial_kernel_params=None, noise_var=None, **kwargs):
        GADDUCBAlgorithm.__init__(self, **kwargs)

        # Config the FEBO domains
        GPConfig.noise_var = noise_var
        # Default is RBF
        if not 'gpy_kernel' in initial_kernel_params:
            initial_kernel_params['gpy_kernel'] = 'GPy.kern.RBF'
        GPConfig.kernels = [(initial_kernel_params['gpy_kernel'], {'variance': initial_kernel_params['variance'], 'lengthscale': initial_kernel_params['lengthscale'] , 'ARD': True})]

        SimpleControllerConfig.T = self.n_iter
        SimpleControllerConfig.best_predicted_every = 1
        
        self.linebo_env = AdaptorBenchmark(self.fn)
        
        _data = []
        for x in self.initial_design:
            y = self.fn(np.array([x]))
            evaluation = np.empty(shape=(), dtype=self.linebo_env.dtype)
            evaluation["x"] = x
            evaluation["y"] = -y
            evaluation["y_exact"] = -y
            evaluation["y_max"] = self.linebo_env._max_value
            
            _data.append(evaluation)
        
        self.initial_data = _data

        # Attempt to return f instead of y if that exist
        self.fn.mlflow_logging.log_init_y(np.min(self.fn.history_y))
        
    def run(self, stopping_Y=None):
        # Setup
        s = None
        try:
            FEBO_Algo = self.FEBO_Algorithm_Cls()
            s = AdaptorController(fn=self.fn, algorithm=FEBO_Algo(), environment=self.linebo_env)

            s.initialize(algo_kwargs = dict(initial_data=self.initial_data))
            s.run(stopping_Y)
        except Exception as e:
            logging.exception("Exception")
        finally:
            if s:
                s.finalize()
    def FEBO_Algorithm_Cls(self):
        raise NotImplementedError

class FEBO_Random(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.Random

class NelderMead(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.NelderMead

class RandomLineBO(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.RandomLineBO

class CoordinateLineBO(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.CoordinateLineBO

class AscentLineBO(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.AscentLineBO

class UCB(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        return febo.algorithms.UCB

class Rembo(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        from febo.algorithms.rembo import Rembo
        return Rembo

class InterleavedRembo(FEBOAlgorithm, metaclass=Algorithm):
    def FEBO_Algorithm_Cls(self):
        from febo.algorithms.rembo import InterleavedRembo
        return InterleavedRembo

class AdaptorController(SimpleController):
    def __init__(self, fn, *args, **kwargs):
        super(AdaptorController, self).__init__(*args, **kwargs)
        self.fn = fn
    def run(self, stopping_Y=None):
        logging.info(f"Starting optimization: {self.algorithm.name}")
        # interaction loop
        while not self._exit:
            self._run_step(stopping_Y=stopping_Y)
            
            evaluation = self._data[-1]
            self.fn.mlflow_logging.log_y(np.min(self.fn.history_y[-1]))

# Random algorithm
class Random(GADDUCBAlgorithm, metaclass=Algorithm):
    def __init__(self, **kwargs):
        GADDUCBAlgorithm.__init__(self, **kwargs)
        self.mlflow_logging = self.fn.mlflow_logging
    def run(self, stopping_Y=None):
        f = self.fn
        initial_design = self.initial_design
        n_iter = self.n_iter
        
        initial_design_iter = self.domain.random_X(self.rs, n_iter)

        Y = []
        Y_best = []
        X_rand = []
        y_best = np.inf
        for x in initial_design:
            y = f(np.array([x]))
            Y.append(y)
            if y < y_best:
                y_best = y
            Y_best.append(y_best)
            X_rand.append(x)

        self.mlflow_logging.log_init_y(np.min(self.fn.history_y))

        for x in initial_design_iter:
            if y_best<=stopping_Y:
                Y.append(y)
                if y < y_best:
                    y_best = y
                Y_best.append(y_best)
                X_rand.append(X_rand[-1])
            else:
                y = f(np.array([x]))
                self.mlflow_logging.log_y(np.min(self.fn.history_y[-1]))
                Y.append(y)
                if y < y_best:
                    y_best = y
                Y_best.append(y_best)
                X_rand.append(x)

            
                

        return Y_best, Y, X_rand

# BayesianOptimization algorithm
class BayesianOptimization(GADDUCBAlgorithm):
    def __init__(self, algorithm_random_seed, lengthscaleNumIter, n_iter, initial_graph=None, initial_kernel_params=None, learnDependencyStructureRate=50,
        learnParameterRate=None, graphSamplingNumIter=100, fully_optimize_lengthscales=False, exploration_weight=2,
        normalize_Y=False, eps=-1, noise_var=0., max_eval=-1, p = 0.5, M=0, max_group_size=0, opt_restart=None, param_exploration=0.1, 
        acq_opt_restarts=1, random_graph=False, learn_random_graph=False, lr=None, random_graph_samples=None, use_baseline=False, kl_penalty=0, iterations=1, ab=False, ab_gamma=0.75, **kwargs):
        GADDUCBAlgorithm.__init__(self, n_iter, algorithm_random_seed, **kwargs)
        self.learnDependencyStructureRate=learnDependencyStructureRate
        self.learnParameterRate = learnParameterRate
        self.graphSamplingNumIter=graphSamplingNumIter
        self.lengthscaleNumIter=lengthscaleNumIter
        self.fully_optimize_lengthscales=fully_optimize_lengthscales
        self.exploration_weight=exploration_weight
        self.normalize_Y=normalize_Y
        self.eps=eps
        self.noise_var=noise_var
        self.max_eval=max_eval
        self.p=p
        self.acq_opt_restarts = acq_opt_restarts
        self.result_path=Config().base_path

        if not hasattr(self, 'random_graph'):
            self.random_graph = False
        
        # Additional Param
        self.exact_feval = False
        # GF should be init here
        # TODO
        dim = self.fn.domain.dimension
        if initial_graph is None:
            if self.random_graph:
                g = get_random_graph(dim, 1)
            else:
                g = nx.empty_graph(dim)
        elif initial_graph == 'full':
            g = nx.complete_graph(dim)
        else:
            g = nx.empty_graph(dim)
            g.add_edges_from(initial_graph)

        self.initial_graph = g
        self.graph_function = self.get_GraphFunction()(self.initial_graph, initial_kernel_params)
        assert(dim == self.graph_function.dimension())
        dim = self.graph_function.dimension()
        if M == 0:
            M = dim
        if max_group_size == 0:
            max_group_size = dim
        self.M=M
        self.max_group_size=max_group_size

        self.opt_restart = opt_restart
        self.param_exploration = param_exploration

        self.kwargs = kwargs

    def run(self, stopping_Y=None):
        try:
            
            mybo = MyBOModular(self.domain, self.initial_design, self.graph_function,
                            max_eval=self.max_eval, fn=self.fn, fn_optimizer=self.make_fn_optimizer(),
                            noise_var=self.noise_var, exact_feval=self.exact_feval, 
                            exploration_weight_function=self.exploration_weight, 
                            learnDependencyStructureRate=self.learnDependencyStructureRate,
                            learnParameterRate=self.learnParameterRate,
                            normalize_Y=self.normalize_Y,
                            acq_opt_restarts=self.acq_opt_restarts,
                            random_graph=self.random_graph,
                            additional_args=self.kwargs)
        except EarlyTerminationException as e:
            self.fn.mlflow_logging.log_cost_ba()
            self.fn.mlflow_logging.log_y(e.metrics['y'])

            while self.fn.mlflow_logging.t_y < self.n_iter:
                self.fn.mlflow_logging.log_cost_ba()
                self.fn.mlflow_logging.log_battack(int(True), self.fn.cnn.target_label[0])
                self.fn.mlflow_logging.log_y(e.metrics['y'])
            return None, None
            
        if self.n_iter > 0:
            try:
                mybo.run_optimization(self.n_iter, eps=self.eps, stopping_Y=stopping_Y)
            except EarlyTerminationException as e:
                
                cost_metrics = self.fn.mlflow_logging.cost_metrics
                ba_metrics = self.fn.mlflow_logging.ba_metrics
                self.fn.mlflow_logging.log_y(e.metrics['y'])

                while self.fn.mlflow_logging.t_y < self.n_iter:
                    
                    self.fn.mlflow_logging.log_cost(cost_metrics['acq_cost'])
                    self.fn.mlflow_logging.log_battack(**ba_metrics)
                    self.fn.mlflow_logging.log_y(e.metrics['y'])

        # np.save(os.path.join(self.result_path,'all_graphs.npy'), mybo.all_graphs)

        return mybo.Y.flatten(), mybo
        
    def FnOptimizer(self):
        raise NotImplementedError
    def make_fn_optimizer(self):
        FnOptimizer = self.FnOptimizer()
        # Update M and max_group_size just in case its not specified
        return FnOptimizer(graphSamplingNumIter=self.graphSamplingNumIter, lengthscaleNumIter=self.lengthscaleNumIter, cycles=self.cycles, 
            fully_optimize_lengthscales=self.fully_optimize_lengthscales, p=self.p, M=self.M, max_group_size=self.max_group_size, sigma2=self.noise_var, 
            opt_restart=self.opt_restart, param_exploration=self.param_exploration)
    def get_GraphFunction(self):
        return GraphFunction

class GraphOverlap(BayesianOptimization, metaclass=Algorithm):
    __metaclass__ = Algorithm
    def __init__(self, **kwargs):
        BayesianOptimization.__init__(self, **kwargs)
    def FnOptimizer(self):
        self.cycles = True
        return function_optimizer.GraphOverlap

class GraphNonOverlap(BayesianOptimization, metaclass=Algorithm):
    __metaclass__ = Algorithm
    def __init__(self, **kwargs):
        BayesianOptimization.__init__(self, **kwargs)
    def FnOptimizer(self):
        self.cycles = True
        return function_optimizer.GraphNonOverlap

class Tree(BayesianOptimization, metaclass=Algorithm):
    __metaclass__ = Algorithm
    def __init__(self, **kwargs):
        BayesianOptimization.__init__(self, **kwargs)
    def FnOptimizer(self):
        self.cycles = False
        return function_optimizer.Tree

class RDUCB(BayesianOptimization, metaclass=Algorithm):
    """
    MIT License

    Copyright (c) 2023 Huawei Technologies Co., Ltd

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    """
    __metaclass__ = Algorithm
    def __init__(self, **kwargs):
        self.random_graph = True
        BayesianOptimization.__init__(self, **kwargs)

    def FnOptimizer(self):
        self.cycles = False
        return function_optimizer.ParameterOnlyOptimizer

class Optimal(BayesianOptimization, metaclass=Algorithm):
    __metaclass__ = Algorithm
    def __init__(self, n_iter, initial_kernel_params, learnDependencyStructureRate, fn, **kwargs):
        self.fn = fn
        # Make sure the fn that it accesses is the true fn without noise
        self.fn.__call__ = self.fn.eval
        logging.info("Ignoring intial_kernel_params and noise_var.")
        # Redefine the inital kernel params to the true kernel
        initial_kernel_params = self.fn.kernel_params
        # n_iter + kwargs['n_rand'] + 10
        # TODO Should take lengthscale from function
        BayesianOptimization.__init__(self, n_iter=n_iter, initial_graph=fn.graph, initial_kernel_params=initial_kernel_params, 
            learnDependencyStructureRate=-1, fn = fn, **kwargs)
        # We also use the optimal lengthscale
        # We also tweak the exportation to be 0
        
        self.noise_var = 0
        self.exact_feval = True
        self.fn.fn_noise_sd = 0
        '''
        self.exploration_weight = 0
        self.noise_var = 0
        self.exact_feval = True
        self.fn.fn_noise_var = 0
        '''
        # The following is a new field

        logging.info("Using True Graph = {}".format(self.initial_graph.edges()))
        logging.info("exploration_weight = {}".format(self.exploration_weight))
        logging.info("noise_var = {}".format(self.noise_var))
        logging.info("exact_feval = {}".format(self.exact_feval))
    def make_fn_optimizer(self):
        return None
    def get_GraphFunction(self):
        return OptimalGraphFunction
