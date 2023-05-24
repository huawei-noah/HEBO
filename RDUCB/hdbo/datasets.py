#!/usr/bin/env python3
import subprocess
import json
import re
import logging
import itertools
import collections
import math
from collections import defaultdict
from itertools import combinations
from functools import partial

import networkx as nx
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pickle
import mlflow
import os.path
import h5py
import json
import sys
import lpsolve_config

from common import Config
from hpolib.benchmarks import synthetic_functions
from GPyOpt.core.task.space import Design_space

def getDecompositionFromGraph(graph):
    cliques = nx.find_cliques(graph)
    decomp = []
    for c in cliques:
        decomp.append(sorted(c))
    return decomp

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
        logging.info("Load loader[%s].", loader_id)
        return MetaLoader.registry[loader_id]

class NAS(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, NAS).__new__(cls, cls_name, bases, attrs)
        NAS.registry[cls_name] = new_class
        NAS.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(bench_type, **kwargs):
        logging.info("Using NAS dataset loader with bench_type[%s].", bench_type)
        return NAS.registry[bench_type]

class Synthetic(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Synthetic).__new__(cls, cls_name, bases, attrs)
        Synthetic.registry[cls_name] = new_class
        Synthetic.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(graph_type, **kwargs):
        logging.info("Using synthetic dataset loader with graph_type[%s].", graph_type)
        return Synthetic.registry[graph_type]

class LPSolve(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, LPSolve).__new__(cls, cls_name, bases, attrs)
        LPSolve.registry[cls_name] = new_class
        LPSolve.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(problem_type, **kwargs):
        logging.info("Using LPSolve problem type[%s].", problem_type)
        return LPSolve.registry[problem_type]

class LassoBenchlib(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, LassoBenchlib).__new__(cls, cls_name, bases, attrs)
        LassoBenchlib.registry[cls_name] = new_class
        LassoBenchlib.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(problem_type, **kwargs):
        logging.info("Using LassoBench problem type[%s].", problem_type)
        return LassoBenchlib.registry[problem_type]

class Hpolib(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Hpolib).__new__(cls, cls_name, bases, attrs)
        Hpolib.registry[cls_name] = new_class
        Hpolib.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(hpo_fn, **kwargs):
        logging.info("Using Hpolib function[%s].", hpo_fn)
        return Hpolib.registry[hpo_fn]

class Simple(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Simple).__new__(cls, cls_name, bases, attrs)
        Simple.registry[cls_name] = new_class
        Simple.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(simple_fn, **kwargs):
        logging.info("Using Simple function[%s].", simple_fn)
        return Simple.registry[simple_fn]

class Domain(Design_space):
    def __init__(self, dimension, combined_domain):
        self.dimension = dimension
        self.combined_domain = combined_domain
        super(Domain, self).__init__(self.get_gpy_domain())
    def get_gpy_domain(self):
        gpy_domain = [{'name': 'x_{}'.format(i), 'type': 'discrete', 'domain': tuple(d), 'dimensionality' : 1 } for i, d in enumerate(self.combined_domain)]
        return gpy_domain
    def get_opt_domain(self):
        space = {}
        space['type'] = 'discrete'
        space['domain'] = self.combined_domain
        return space
    def none_value(self):
        return np.array([-1] * self.dimension, dtype=np.float)
    def random_X(self, rs, n_rand):
        # Pick from each dimension's domain 
        X_T = []
        for ea_d in self.combined_domain:
            X_T.append(rs.choice(ea_d, n_rand, replace=True))
        return np.array(X_T).T
        
class SyntheticDomain(Domain):
    # fingerprints are the binary version of the fingerprints output by the Chem package
    def __init__(self, dimension, grid_size, domain_lower, domain_upper):
        self.grid_size = grid_size
        
        # The actual discretized domain in any dimension
        self.X_domain = np.linspace(domain_lower, domain_upper, grid_size)
        self.index_domain = list(range(self.grid_size))
        super(SyntheticDomain, self).__init__(dimension, [self.X_domain] * dimension)
    def generate_grid(self, dim):
        # This generates a N-Dim Grid
        return np.array(np.meshgrid(*[self.X_domain] * dim)).T.reshape(-1, dim)
    def translate(self, X_indices):
        return self.X_domain[X_indices]

from functools import reduce
# Barebones component function
class ComponentFunction(dict):
    def __init__(self, fn_decomp_lookup):
        self.__dict__ = fn_decomp_lookup
        
    def __call__(self, x):
        # Does not matter the sequence
        '''
        for ea_cfn_f in self.__dict__.values():
            f_parts.append(ea_cfn_f(x))
        return np.sum(f_parts,axis=0)
        '''
        #return np.array([ ea_cfn(x) for ea_cfn in self.__dict__.values() ]).sum(axis=0)
        return reduce(np.add, map(lambda ea: ea(x), self.__dict__.values()))
    def acq_f_df(self, x):
        
        f_parts = []
        g_parts = []
        for ea_cfn in self.__dict__.values():
            f_part, g_part = ea_cfn.acquisition_function_withGradients(x)
            f_parts.append(f_part)
            g_parts.append(g_part)
        return np.sum(f_parts,axis=0), np.sum(g_parts,axis=0)
        #return np.array([ ea_cfn.acquisition_function_withGradients(x) for ea_cfn in self.__dict__.values() ]).sum(axis=0)
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    def __getitem__(self, key):
        return self.__dict__[key]
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    def has_key(self, k):
        return k in self.__dict__
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def pop(self, *args):
        return self.__dict__.pop(*args)
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __unicode__(self):
        return unicode(repr(self.__dict__))


# Synthetic Component function
class SyntheticComponentFunction(ComponentFunction):
    def __init__(self, graph, fn_decomp_lookup):

        # Find all maximal cliques
        cfns = [ tuple(sorted(c_vertices)) for c_vertices in nx.find_cliques(graph)] 
        cfns = sorted(cfns, key=lambda x : -len(x))

        all_fns = set()
        
        cfn_dict = {}
        for cfn in cfns:
            if len(cfn) == 1:
                cfn_decomposition = set( [ fn_decomp_lookup[cfn] ] )
            else:
                cfn_decomposition = set( [ fn_decomp_lookup[tuple(sorted(edge))] for edge in combinations(cfn, 2) ])

            # Make sure we do not have repeated cliques, or empty functions
            cfn_decomposition = cfn_decomposition - all_fns
            if len(cfn_decomposition) == 0:
                continue

            all_fns.update(cfn_decomposition)

            
            def cfn_eval(x, _cfn_decomposition = cfn_decomposition):
                return np.array([ ea_cfn(x) for ea_cfn in _cfn_decomposition ]).sum(axis=0)

            cfn_dict[cfn] = cfn_eval
        super(SyntheticComponentFunction, self).__init__(cfn_dict)

class Function(object):
    def __init__(self, domain):
        self.domain = domain
        self.history_y = []
    def eval(self, x):
        raise NotImplementedError
    def __call__(self, x):
        # This call is with noise added
        y = self.eval(x)
        self.history_y.append(y)
        return y
    def get_emb_dim(self):
        return max(2, int(np.sqrt(self.domain.dimension)))
    def has_synthetic_noise(self):
        return False

class NoisyFunction(Function):
    def __init__(self, domain, rs, fn_noise_var):
        Function.__init__(self, domain)
        self.rs = rs
        self.fn_noise_sd = np.sqrt(fn_noise_var)
    def __call__(self, x):
        y = self.eval(x)
        self.history_y.append(y)
        return self.rs.normal(0, self.fn_noise_sd) + y
    def has_synthetic_noise(self):
        return True

# Function for NAS test
class ConfigLosses(Function):
    # fingerprints are the binary version of the fingerprints output by the Chem package

    def __init__(self, parameters, key_map, domain, data, rs):
        Function.__init__(self, domain)
        self.parameters = parameters
        self.dim = len(parameters)
        self.data = data
        self.key_map = key_map
        self.rs = rs
        self.graph = None
    def eval(self, x):
        x = x[0]

        index = self.rs.randint(4)
        config_dict = {}
        for i in range(len(self.parameters)):
            config_dict[self.parameters[i]] = self.key_map[i][x[i]]
        k = json.dumps(config_dict, sort_keys=True)
        valid = self.data[k]["valid_mse"][index]
        return valid[-1]

# https://stackoverflow.com/questions/47370718/indexing-numpy-array-by-a-numpy-array-of-coordinates
def ravel_index(b, shp):
    return np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1],[1])).dot(b)

# Synthetic
class FunctionValues(NoisyFunction):
    # fingerprints are the binary version of the fingerprints output by the Chem package
    def __init__(self, f_list, v_list, domain, fn_decomposition, graph, kernel_params, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.f_list = f_list
        self.v_list = v_list
        self.fn_decomposition = fn_decomposition
        # True graph and true lengthscale
        self.graph = graph
        self.kernel_params = kernel_params
        self.v_flat = [ list(v) for v in v_list ]
    def eval(self, x):
        x_i = np.searchsorted(self.domain.X_domain, x)
        # This actually evaluates the function
        return self.eval_indexed(x_i)
    def eval_indexed(self, x_i):
        return sum([ self._part_eval(ea_v, ea_f, x_i) for ea_v, ea_f in zip(self.v_flat, self.f_list) ])
    def part_eval(self,index_f, x):
        x_i = np.searchsorted(self.domain.X_domain, x)
        return [[ self._part_eval(self.v_flat[index_f], self.f_list[index_f], ea_x_i) ] for ea_x_i in x_i ]
    # Evaluate only that edge, internal function
    def _part_eval(self, ea_v, ea_f, x):
        return np.take(ea_f, ravel_index(np.take(x, ea_v), ea_f.shape))
    def make_component_function(self):
        fn_decomp_lookup = {}
        for i, decomp in enumerate(self.fn_decomposition):
            fn_decomp_lookup[decomp] = partial(self.part_eval, i)
        return SyntheticComponentFunction(self.graph, fn_decomp_lookup)

class Loader(object):
    def __init__(self, dataID, hash_data, **kwargs):
        self.dataID = dataID
        self.kwargs = kwargs
        self.hash_data = hash_data
    def get_dataset_id(self):
        return self.__class__.__name__ + self.dataID
    def load(self):
        # Save the ground truth network, if it exist
        self.log_true_graph()

        cached_file_path = self.cached_file_path()
        if os.path.isfile(cached_file_path):
            logging.info("Loading pre-computed function at {}.".format(cached_file_path))
            with open(cached_file_path, 'rb') as handle:
                fn, soln = pickle.load(handle)
                if isinstance(self, NetworkxGraph):
                    logging.info("Checking consistency of pre-compute.")
                    assert(nx.is_isomorphic(fn.graph, self.get_nx_graph()) )

            # Super hacks, for compatibility purposes
            # TODO
            if not hasattr(fn.domain, 'model_dimensionality'):
                super(Domain, fn.domain).__init__(fn.domain.get_gpy_domain())

            return fn, soln
        else:
            logging.info("No pre-computed function, computing.")
            return self._load(), None
    def cached_file_path(self):
        home_dir = self.kwargs["home_dir"] if 'home_dir' in self.kwargs else "~"
        return Config(home_dir=home_dir).cache_file('{}.pkl'.format(self.hash_data))
    def save(self, fn, soln):
        cached_file_path = self.cached_file_path()
        logging.info("Saving pre-computed function at {}.".format(cached_file_path))
        with open(cached_file_path, 'wb') as handle:
            pickle.dump((fn, soln), handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _load(self):
        raise NotImplementedError
    def log_true_graph(self):
        pass

class NASLoader(Loader):
    def __init__(self, dataID, dimension, data_random_seed, hyper_values, key_map, parameters, **kwargs):
        Loader.__init__(self, '{}Nas-DRS{}-D{}'.format(dataID, data_random_seed, dimension), **kwargs)
        self.rs = np.random.RandomState(data_random_seed)
        self.hyper_values = hyper_values
        self.key_map = key_map
        self.parameters = parameters
        self.dimension = dimension
    def load(self):
        tabular_benchmark_path = self.tabular_benchmark_path()
        # No precompute, we load directly
        if os.path.isfile(tabular_benchmark_path):
            logging.info("Found fcnet benchmark at {}".format(tabular_benchmark_path))
        else:
            logging.fatal("Required fcnet benchmark file - {} is not found".format(tabular_benchmark_path))

        data = h5py.File(tabular_benchmark_path, "r")

        best_k = None
        best_validation_error = np.inf
        for k in data.keys():
            validation_error = np.min(data[k]["valid_mse"][:, -1])
            if best_validation_error > validation_error:
                best_k = k
                best_validation_error = validation_error
                
        soln = (best_k, best_validation_error, len(data.keys()))
        config_losses = ConfigLosses(parameters=self.parameters, key_map=self.key_map, domain=Domain(self.dimension, self.hyper_values), data=data, rs=self.rs)

        return config_losses, soln

class FcnetLoader(NASLoader, metaclass=NAS):
    def __init__(self, fcnet_filename, **kwargs):
        dimension = 9
        hyper_values = [
            np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]),
            np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.125, 0.25, 0.5, 1.0])
        ]
        key_map = [
            {0.03125:16, 0.0625:32, 0.125:64, 0.25:128, 0.5:256, 1.0:512},
            {0.03125:16, 0.0625:32, 0.125:64, 0.25:128, 0.5:256, 1.0:512},
            {0.0:0.0, 0.5:0.3, 1.0:0.6},
            {0.0:0.0, 0.5:0.3, 1.0:0.6},
            {0.0: 'relu', 1.0: 'tanh'},
            {0.0: 'relu', 1.0: 'tanh'},
            {0.005:0.0005, 0.01:0.001, 0.05:0.005, 0.1:0.01, 0.5:0.05, 1.0:0.1},
            {0.0: 'cosine', 1.0: 'const'},
            {0.125:8, 0.25:16, 0.5:32, 1.0:64}
        ]
        parameters = ["n_units_1", "n_units_2", "dropout_1", "dropout_2", "activation_fn_1", "activation_fn_2", "init_lr", "lr_schedule", "batch_size"]
        NASLoader.__init__(self, "", dimension=dimension, hyper_values=hyper_values, key_map=key_map, parameters=parameters, **kwargs)
        self.fcnet_filename = fcnet_filename
    def tabular_benchmark_path(self):
        home_dir = self.kwargs["home_dir"] if 'home_dir' in self.kwargs else "~"
        return os.path.join(home_dir, "fcnet/{}.hdf5".format(self.fcnet_filename))

class SyntheticLoader(Loader):
    def __init__(self, dataID, dimension, kernel_params, data_random_seed, grid_params, fn_noise_var, **kwargs):
        
        self.fn_noise_var = fn_noise_var
        self.kernel_params = kernel_params

        # Unpack kernel parameter for easy use
        lengthscale = kernel_params["lengthscale"]
        variance = kernel_params["variance"]

        grid_size = grid_params["grid_size"]
        domain_lower = grid_params["domain_lower"]
        domain_upper = grid_params["domain_upper"]

        Loader.__init__(self, '{}Syn-DRS{}-D{}-Grid{}[{},{}]-L{}V{}'.format(dataID, data_random_seed, dimension, grid_size, domain_lower, domain_upper, lengthscale, variance), **kwargs)
        self.dimension = dimension
        self.rs = np.random.RandomState(data_random_seed)
        self.domain = SyntheticDomain(dimension, grid_size, domain_lower, domain_upper)

        # We will not compute the functions as per cliques as its computationally intractable
        self.cliques = list(nx.find_cliques(self.get_nx_graph()))
        
        # We will instead decompose it to 1D and 2D functions
        # TODO REFACTOR
        self.fn_decomposition = [ tuple(sorted([v])) for v in nx.isolates(self.get_nx_graph())] + [ tuple(sorted(e)) for e in self.get_nx_graph().edges() ]

        # Lengthscale belongs to each function
        # ground truth lengthscale
        self.lengthscale = lengthscale
        if type(lengthscale) == float or type(lengthscale) == int:
            self.dimensional_lengthscale = [lengthscale] * len(self.fn_decomposition)
        else:
            raise NotImplementedError

        self.variance = variance
        if type(variance) == float or type(variance) == int:
            self.dimensional_variance = [variance] * len(self.fn_decomposition)
        else: 
            raise NotImplementedError

    def get_nx_graph(self):
        raise NotImplementedError
    def generate_functions(self):
        # Group GPs by dimension and lengthscale
        # We do this so we can compute really quickly
        dim_ls_dict = defaultdict(list)
        for v, ls, variance in zip(self.fn_decomposition, self.dimensional_lengthscale, self.dimensional_variance):
            dim_ls_dict[( len(v), ls, variance )].append(v)

        v_list = []
        f_list = []
        for k in dim_ls_dict:
            v_dim, ls, variance = k
            variables = dim_ls_dict[k]
            f_list += list(self.generate_functions_same_distribution(len(variables), ls, variance, v_dim))
            v_list += list(variables)

        # Generate for all length 2, with given lengthscale
        return f_list, v_list
    # Generates n_functions of GP with dim dimensions with the same lengthscale
    def generate_functions_same_distribution(self, n_functions, lengthscale, variance, v_dim):
        N = self.domain.grid_size
        grid = self.domain.generate_grid(v_dim)
        ker = GPy.kern.RBF(input_dim=v_dim, lengthscale=lengthscale, variance=variance)
        mu = np.zeros(N**v_dim) #(N*N)
        C = ker.K(grid, grid) #(N*N)
        # The following function will generate n_functions * (N*N)
        fun = self.rs.multivariate_normal(mu, C, (n_functions), check_valid='raise')
        target_shape = (n_functions,) + (N,) * v_dim
        # Which will need to be reshaped to n_functions * N * N
        return fun.reshape(target_shape)
    def log_true_graph(self):
        nx.draw(self.get_nx_graph(), cmap = plt.get_cmap('jet'), with_labels=True)
        plt.savefig(Config().data_file('ground_truth_graph.png'))
        plt.clf()
    def _load(self):
        f_list, v_list = self.generate_functions()
        self.function = FunctionValues(f_list, v_list, self.domain, self.fn_decomposition, self.get_nx_graph(), self.kernel_params, self.rs, self.fn_noise_var)

        return self.function

class NetworkxGraph(SyntheticLoader):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(nx.convert_node_labels_to_integers(G))
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)
    def make_graph(self, dimension):
        raise NotImplementedError
    def get_nx_graph(self):
        return self.true_dependency_graph

class EmptyGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.empty_graph(dimension)

class PathGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.path_graph(dimension)

class TestGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.random_partition_graph([1,1,2,2,3],1,0)
        
class ErdosRenyiGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.erdos_renyi_graph(dimension, np.random.RandomState(self.data_random_seed).rand(), seed=self.data_random_seed)

class StarGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        #Return the Star graph with n+1 nodes: one center node, connected to n outer nodes.
        return nx.star_graph(dimension-1)

class GridGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        n = int(np.sqrt(dimension))
        # Ensure perfect square
        assert(np.isclose(n**2, dimension))
        return nx.grid_2d_graph(n, n)

class GridLargeGraph(GridGraph, metaclass=Synthetic):
    def load(self):
        # Save the ground truth network, if it exist
        self.log_true_graph()

        cached_file_path = self.cached_file_path()
        if os.path.isfile(cached_file_path):
            logging.info("Loading pre-computed function at {}.".format(cached_file_path))
            with open(cached_file_path, 'rb') as handle:
                fn, soln = pickle.load(handle)
                if isinstance(self, NetworkxGraph):
                    logging.info("Checking consistency of pre-compute.")
                    assert(nx.is_isomorphic(fn.graph, self.get_nx_graph()) )

            return fn, soln
        else:
            logging.info("Pre-computed function has no answer")
            fn, soln = super().load()
            soln = (None, 0, 0)
            self.save(fn, soln)
            return fn, soln

class GridGraph34(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        assert(dimension == 12)
        return nx.grid_2d_graph(4, 3)

class PartitionGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        assert(dimension == 12)
        return nx.random_partition_graph([3,3,3,3],1,0)

class SparseErdosRenyiGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.erdos_renyi_graph(dimension, 2.0/dimension, seed=self.data_random_seed)

class PowerlawTree(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        sys.setrecursionlimit(1500)
        return nx.random_powerlaw_tree(dimension, seed=self.data_random_seed, tries=dimension**2)

class AncestryGraph(NetworkxGraph, metaclass=Synthetic):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(G)
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)

    def make_graph(self, dimension):
        # recursion limit
        assert(dimension == 132)
        G, self.shells = pickle.load(open("data/ancestry.pkl", 'rb'))
        return G
    def log_true_graph(self):
        # plt.rcParams['figure.figsize'] = [15, 15]
        pos = nx.shell_layout(self.get_nx_graph(), self.shells)
        nx.draw(self.get_nx_graph(), cmap = plt.get_cmap('jet'), with_labels=True, pos=pos)
        plt.savefig(Config().data_file('ground_truth_graph.png'))
        plt.clf()

class DebugGraph(NetworkxGraph, metaclass=Synthetic):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(G)
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)
    def make_graph(self, dimension):
        # recursion limit
        G = nx.read_gpickle("graph.pkl")
        assert(dimension == len(G.nodes()))
        return G

# =======================================================================
class MpsLoader(Loader, metaclass=LPSolve):
    def __init__(self, mps_filename, infinite, time_limit, max_floor, **kwargs):
        
        self.dimension = lpsolve_config.dimension

        self.hyper_values = lpsolve_config.hyper_values
        self.key_map = lpsolve_config.key_map
        self.parameters = lpsolve_config.parameters
        
        self.mps_filename = mps_filename
        self.infinite = infinite
        self.time_limit = time_limit
        self.max_floor = max_floor
        Loader.__init__(self, '{}-Mps-D{}'.format(mps_filename, self.dimension), **kwargs)
    def mps_path(self):
        home_dir = self.kwargs["home_dir"] if 'home_dir' in self.kwargs else "~"
        return os.path.join(home_dir, "mps/{}.mps".format(self.mps_filename))
    def load(self):
        mps_path = self.mps_path()
        # No precompute, we load directly
        if os.path.isfile(mps_path):
            logging.info("Found MPS File at {}".format(mps_path))
        else:
            logging.fatal("Required MPS File - {} is not found".format(mps_path))

        config_losses = ExecuteLPSolve(parameters=self.parameters, key_map=self.key_map, domain=Domain(self.dimension, self.hyper_values), mps_path=mps_path, infinite=self.infinite, time_limit=self.time_limit, max_floor=self.max_floor)

        return config_losses, (None, 0, 0)

# Function to execute LP Solve
class ExecuteLPSolve(Function):
    def __init__(self, parameters, key_map, domain, mps_path, infinite, time_limit, max_floor):
        Function.__init__(self, domain)
        self.parameters = parameters
        self.dim = len(parameters)
        self.mps_path = mps_path
        self.key_map = key_map
        self.graph = None
        self.infinite = infinite
        self.time_limit = time_limit
        self.max_floor = max_floor
    def eval(self, x):

        args = json.dumps({
            "x":x.tolist(),
            "mps_path":self.mps_path,
            "infinite":self.infinite,
            "time_limit":self.time_limit})
        
        #os.system('python ./hdbo/lpsolve.py \'{}\' '.format(args))
        cmd = 'python ./hdbo/lpsolve.py \'{}\' '.format(args)
        logging.debug(cmd)
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, timeout=self.time_limit*5)
            logging.info(output)
            obj_val_str = re.findall(r"RETURN_OBJECTIVE_VALUE:\((\d*(?:\.\d*)?(?:e[\+|-]{0,1}\d+){0,1})\)", str(output))
            objective = min(float(obj_val_str[0]), self.max_floor)
        except Exception as e:
            logging.exception("LPSOLVE Exception")
            logging.error("cmd: {}".format(cmd))
            objective = self.max_floor
        
        logging.info("Objective Value: {}".format(objective))
        
        return objective
# =======================================================================
class LassoBenchLoader(Loader, metaclass=LassoBenchlib):
    """
    MIT License

    Copyright (c) 2023  Huawei Technologies Co., Ltd

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
    def __init__(self, pick_data, grid_size, fidelity, fixed_dims=0, **kwargs):
        
        self.f = ExecuteLassoBench(pick_data, grid_size, fidelity, fixed_dims)

    def load(self):

        return self.f, (None, 0, 0)

# Function to execute LP Solve
class ExecuteLassoBench(Function):
    """
    MIT License

    Copyright (c) 2023  Huawei Technologies Co., Ltd

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
    def __init__(self, pick_data, grid_size, fidelity=0, fixed_dims=0):
        import LassoBench

        self.f = LassoBench.RealBenchmark(pick_data=pick_data, mf_opt='discrete_fidelity')

        incr = 2 / (grid_size + 1)

        active_dim = self.f.n_features - fixed_dims

        self.hyper_values = [[-1 + incr*ix for ix in range(int(2/incr))] for _ in range(active_dim)]
        self.f = LassoBench.RealBenchmark(pick_data=pick_data, mf_opt='discrete_fidelity')
        domain = Domain(active_dim, self.hyper_values)
        self.parameters = [i for i in range(active_dim)]
        self.dim = len(self.parameters)
        self.graph = None
        self.fidelity = fidelity
        self.append_chunk = [0 for _ in range(fixed_dims)]

        Function.__init__(self, domain)

    def eval(self, x):
        inp = np.array(np.append(x[0], self.append_chunk))
        out = self.f.fidelity_evaluate(inp, index_fidelity=self.fidelity)

        logging.info("Objective Value: {}".format(out))

        return out
# =======================================================================
class HpolibLoader(Loader):
    def __init__(self, data_random_seed, grid_size, fn_noise_var, **kwargs):
        Loader.__init__(self, "Hpolib", **kwargs)
        self.hpo_fn = self.make_hpo_fn()
        self.grid_size = grid_size
        self.rs = np.random.RandomState(data_random_seed)
        self.fn_noise_var = fn_noise_var
    def load(self):
        
        info = self.hpo_fn.get_meta_information()
        soln = (info['optima'], info['f_opt'], None)
        domain = HpoDomain(self.grid_size, info['bounds'])
        hpo_fn_wrapper = HpolibWrapper(domain, self.hpo_fn, self.rs, self.fn_noise_var)
        
        return hpo_fn_wrapper, soln

# Function to execute LP Solve
class HpolibWrapper(NoisyFunction):
    def __init__(self, domain, hpo_fn, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.hpo_fn = hpo_fn
        self.graph = None
    def eval(self, X):
        x = X[0]
        return self.hpo_fn(x)

class Adversarial(Loader, metaclass=Simple):
    """
    MIT License

    Copyright (c) 2023  Huawei Technologies Co., Ltd

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
    def __init__(self, grid_size, **kwargs):
        Loader.__init__(self, "Adversarial", **kwargs)
        logging.info("Using Adversarial function")
        self.domain = SyntheticDomain(3, grid_size, 0, 1.0)
        self.soln = (np.array([0.8, 0.8, 0]), -3.437432141220581, None)
    def load(self):
        return AdversairalFunction(self.domain), self.soln

class AdversairalFunction(Function):
    """
    MIT License

    Copyright (c) 2023  Huawei Technologies Co., Ltd

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
    def __init__(self, domain):
        Function.__init__(self, domain)
        self.graph = None
    def eval(self, X):
        LANDMARKS = [(0.8, 0.8), (0.3, 0.3)]
        WEIGHTS = [1, 2.5]
        COVS = [[0.02,0.02,0.75], [0.01,0.01,0]]
        def f(x, y):
            val = 0
            sum_w = sum(WEIGHTS)
            for i in range(1):
                varx, vary, covxy = COVS[i]
                mux, muy = LANDMARKS[i]
                val += WEIGHTS[i]/sum_w *  1/(2*np.pi * np.sqrt(varx * vary * (1 - covxy**2)) ) * np.exp(( 
                    - 0.5 * (x - mux)**2 / varx - 0.5 * (y - muy)**2 / vary + (x - mux)*(y - muy) * covxy /np.sqrt(vary*varx)
                ) / (1 - covxy**2))

            val += WEIGHTS[1]/sum_w *  1/(2*np.pi * np.sqrt(COVS[1][0])) * np.exp(( 
                    - 0.5 * (x - LANDMARKS[1][0])**2 / COVS[1][0]
                ))
            val += WEIGHTS[1]/sum_w *  1/(2*np.pi * np.sqrt(COVS[1][1])) * np.exp(( 
                    - 0.5 * (y - LANDMARKS[1][1])**2 / COVS[1][1]
                ))

            return -val

        logging.info(f"Querying {X[0][:2]}")
        return np.array([f(X[i][0], X[i][1]) for i in range(len(X))])

    def get_default_values(self):
        values = []
        for i in [0, 1]:
            values += [[0, 0, i], [0.6, 0.6, i], [0, 0.6, i], [0.6, 0, i], [0.2, 0.6, i], [0.6, 0.2, i], [0.4, 0.6, i], [0.6, 0.4, i], [0.2, 0.2, i], [0.4, 0.4, i], [0.4, 0.2, i], [0.2, 0.4, i]]
        return np.array(values)

class HpoDomain(SyntheticDomain):
    # Note that it does not work well when the grid too uneven
    # Uniformly distribute the grid up
    def __init__(self, grid_size, hpo_bounds):
        self.grid_size = grid_size
        dimension = len(hpo_bounds)
        
        joint_domain = []
        all_lower, all_upper = hpo_bounds[0]
        for domain_lower, domain_upper in hpo_bounds:
            joint_domain.append(np.linspace(domain_lower, domain_upper, grid_size))

        self.index_domain = list(range(self.grid_size))
        Domain.__init__(self, dimension, joint_domain)

# Some popular functions
# ========================
class Rosenbrock20D(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.rosenbrock.Rosenbrock20D()

class Hartmann6(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Hartmann6()

class Camelback(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Camelback()

# Some popular functions
# ========================

# ================================================================================
# Permute the domain but
class HpolibAugLoader(HpolibLoader):
    def __init__(self, aug_dimension, **kwargs):
        HpolibLoader.__init__(self, **kwargs)
        self.aug_dimension = aug_dimension

        info = self.hpo_fn.get_meta_information()
        self.actual_dimension = len(info['bounds'])
        total_dimension = self.actual_dimension + self.aug_dimension

        # Compute the permutations
        self.per = self.rs.permutation(total_dimension)
        self.inv_per = np.argsort(self.per)

    def load(self):
        
        info = self.hpo_fn.get_meta_information()

        opt_x = np.concatenate([info['optima'][0], np.zeros(self.aug_dimension)])[self.per]
        soln = (opt_x, info['f_opt'], None)

        # This is fine because the bounds are uniform
        bounds = info['bounds']
        lowers, uppers = zip(*bounds)

        self.aug_lower = min(lowers)
        self.aug_upper = max(uppers)

        bounds = bounds + [ [self.aug_lower, self.aug_upper] for i in range(self.aug_dimension) ]
        bounds = np.array(bounds)
        
        domain = HpoDomain(self.grid_size, bounds[self.per])
        hpo_fn_wrapper = HpolibAugWrapper(domain, self.actual_dimension, self.inv_per, self.hpo_fn, self.rs, self.fn_noise_var)
        
        return hpo_fn_wrapper, soln

class Hartmann6Aug(HpolibAugLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Hartmann6()

class CamelbackAug(HpolibAugLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Camelback()

# Function to execute LP Solve
class HpolibAugWrapper(HpolibWrapper):
    def __init__(self, domain, actual_dimension, inv_per, hpo_fn, rs, fn_noise_var):
        HpolibWrapper.__init__(self, domain, hpo_fn, rs, fn_noise_var)
        self.inv_per = inv_per
        self.actual_dimension = actual_dimension
    def eval(self, X):
        X = X[0] # Compatibility
        X = X[self.inv_per]  # undo permutation
        X = X[:self.actual_dimension]  # take active dimensions
        return self.hpo_fn(X)
    def get_emb_dim(self):
        return self.actual_dimension

# ================================================================================
class Gaussian(Loader, metaclass=Simple):
    def __init__(self, dimension, grid_size, initial_value, **kwargs):
        Loader.__init__(self, "Gaussian", **kwargs)
        self.domain = SyntheticDomain(dimension, grid_size, -1.0, 1.0)
        self.initial_value = initial_value
        self.soln = (np.zeros(dimension), -1.0, None)
    def load(self):
        return SimpleSyntheticFn(self.domain), self.soln

class SimpleSyntheticFn(Function):
    def __init__(self, domain):
        Function.__init__(self, domain)
        self.graph = None
    def eval(self, X):
        X = np.atleast_2d(X[0])
        Y = np.exp(-4*np.sum(np.square(X), axis=1))[0]
        return -Y

class Stybtang(Loader, metaclass=Simple):
    def __init__(self, dimension, grid_size, data_random_seed, fn_noise_var, **kwargs):
        Loader.__init__(self, "Stybtang", **kwargs)
        self.rs = np.random.RandomState(data_random_seed)
        self.fn_noise_var = fn_noise_var
        self.domain = SyntheticDomain(dimension, grid_size, -4.0, 4.0)
        self.soln = (np.array([ -2.903534 ] * dimension), -39.16599*dimension, None)
    def load(self):
        return StybtangFn(self.domain, self.rs, self.fn_noise_var), self.soln

class StybtangFn(NoisyFunction):
    def __init__(self, domain, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.graph = None
    def eval(self, X):
        X = np.atleast_2d(X[0])
        Y = np.sum(X**4 -16.*X**2 + 5.*X, axis=1)/2.
        return Y

class Rosenbrock(Loader, metaclass=Simple):
    def __init__(self, dimension, grid_size, data_random_seed, **kwargs):
        Loader.__init__(self, "Rosenbrock", **kwargs)
        self.rs = np.random.RandomState(data_random_seed)
        self.d = dimension
        self.domain = SyntheticDomain(dimension, grid_size, -5.0, 10.0)
        self.soln = (np.array([ 1 ] * dimension), 0, None)
    def load(self):
        return RosenbrockFn(self.d, self.domain), self.soln

class RosenbrockFn():
    def __init__(self, dimensions, domain):
        self.domain = domain
        self.graph = None
        self.d = dimensions
        self.history_y = []
    def __call__(self, x):
        y = self.eval(x)
        self.history_y.append(y)
        return  y
    def has_synthetic_noise(self):
        return False
    def eval(self, x):
        y = 0
        for i in range(self.d - 1):
            y += 100 * (x[:,i + 1] - x[:,i] ** 2) ** 2
            y += (x[:,i] - 1) ** 2
        return y
