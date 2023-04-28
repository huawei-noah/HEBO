import math
import matplotlib.pyplot as plt
from graph_utils import get_random_graph, sigmoid
from common import Config
from disjoint_set import DisjointSet
from itertools import combinations, product 
import itertools
import numpy as np
import networkx as nx
import GPy
import logging
import random
import scipy
from myAcquisitionLCB import MyAcquisitionLCB
from datasets import ComponentFunction, SyntheticComponentFunction
from functools import partial
from GPy.inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from GPy.models.gp_regression import likelihoods
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
from functools import lru_cache
# Alternative computation using sklearn's kernels
import sklearn.gaussian_process
from graph_utils import get_random_graph

from functools import lru_cache

class KernelWrap(object):
    def __init__(self, kernel, data):
        self.kernel = kernel
        self.active_dims = kernel.active_dims
        self.data = data
    def __eq__(self, other):
        return (self.__hash__() == other.__hash__())
    def __hash__(self):

        ls_wrap = self.kernel.lengthscale
        if type(self.kernel.lengthscale) == np.ndarray:
            ls_wrap = tuple(self.kernel.lengthscale)
        
        return hash((id(self.kernel), tuple(self.kernel.active_dims), self.kernel.variance, ls_wrap))
        
    def __call__(self, *args):
        return self.kernel(*args)

@lru_cache(maxsize=1024)
def cached_apply_X(kernel):
    return kernel(kernel.data.X[:,kernel.active_dims])


#from profilehooks import profile

# Optimizes a single GP kernel function with parameters
class KernelOptimizer(object):
    pass

# Specifically for RBF Kernel
class RBFOptimizer(KernelOptimizer):
    pass

# Optimizes the entire graph structure
class FunctionOptimizer(object):
    pass

class GraphOptimizer(object):
    def __init__(self, graphSamplingNumIter, lengthscaleNumIter, cycles, fully_optimize_lengthscales, p, M, max_group_size, sigma2, opt_restart, param_exploration):
        self.graphSamplingNumIter = graphSamplingNumIter

        self.cycles = cycles
        self.fully_optimize_lengthscales = fully_optimize_lengthscales
        
        self.p = p
        self.M = M
        self.max_group_size = max_group_size
        #self.sigma2 = 1e-8
        self.sigma2 = sigma2 + 1e-8

        assert(self.M!=0)
        assert(self.max_group_size!=0)
        self.context = Context(self.fully_optimize_lengthscales, self.p, self.M, self.max_group_size, self.sigma2, self.cycles, lengthscaleNumIter, opt_restart, param_exploration)

        # TODO some hack
        self.my_learnt_graph_iter_count = 0
    def optimize(self, X, Y_vect, graph_function):

        if self.context.opt_restart==None:
            # Normal mode
            cached_apply_X.cache_clear()
            # gen_candidates=20 ??
            h = self.HypothesisType()(graph_function, self.context, Data(X, Y_vect))
            #print(h.dimensional_parameters)
            #h.evaluate()
            h.likelihood = h._compute_dataLogLikelihood(groupSize=-1, alpha=-1)
            #h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
            h = self._optimize_hypotheses(h)
        else:
            dimensional_parameters = np.array(graph_function.dimensional_parameters)
            
            hypothesis_candidates = []
            # Beast mode, opt restarts
            for opt_i in range(self.context.opt_restart):
                _perturbed_dim_param = np.random.normal(dimensional_parameters, dimensional_parameters*self.context.param_exploration)
                _perturbed_dim_param = np.array(list(map(self.context.kern_respect_bounds, _perturbed_dim_param)))
                #print(_perturbed_dim_param)
                graph_function.dimensional_parameters = _perturbed_dim_param

                # Normal mode
                cached_apply_X.cache_clear()
                # gen_candidates=20 ??
                h = self.HypothesisType()(graph_function, self.context, Data(X, Y_vect))
                #print(h.dimensional_parameters)
                #h.evaluate()
                h.likelihood = h._compute_dataLogLikelihood(groupSize=-1, alpha=-1)
                #h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
                h = self._optimize_hypotheses(h)
        
                best_graph = h.make_graph()
                logging.info("Candidate graph : {} - {}".format(h.likelihood, best_graph.edges()))
                
                hypothesis_candidates.append(h)


            # Pick the best
            h = max(hypothesis_candidates, key=lambda h: h.likelihood)
            #print(h.likelihood)

        best_graph = h.make_graph()
        best_dim_params = h.dimensional_parameters

        nx.draw(best_graph, cmap = plt.get_cmap('jet'), with_labels=True)
        plt.savefig(Config().learnt_graphs_file('{:05d}.png'.format(self.my_learnt_graph_iter_count)))
        plt.clf()
        self.my_learnt_graph_iter_count += 1
        logging.info("New graph : {}".format(best_graph.edges()))

        # Perform the update
        graph_function.graph = best_graph
        graph_function.dimensional_parameters = best_dim_params
        graph_function.kernels = graph_function._make_kernels(best_dim_params, graph_function.make_fn_decompositions())
        
        return h.likelihood

    # Update only parameters
    def optimize_parameters(self, X, Y_vect, graph_function):
        
        if self.context.opt_restart==None:
            # Normal mode
            h = self.HypothesisType()(graph_function, self.context, Data(X, Y_vect))
            #print(h.dimensional_parameters)
            #h.evaluate()
            h.likelihood = h._compute_dataLogLikelihood(groupSize=-1, alpha=-1)
            h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
        else:
            dimensional_parameters = np.array(graph_function.dimensional_parameters)
            
            hypothesis_candidates = []
            # Beast mode, opt restarts
            for opt_i in range(self.context.opt_restart):

                _perturbed_dim_param = np.random.normal(dimensional_parameters, dimensional_parameters*self.context.param_exploration)
                _perturbed_dim_param = np.array(list(map(self.context.kern_respect_bounds, _perturbed_dim_param)))
                #print(_perturbed_dim_param)
                graph_function.dimensional_parameters = _perturbed_dim_param

                # Normal mode
                h = self.HypothesisType()(graph_function, self.context, Data(X, Y_vect))
                #print(h.dimensional_parameters)
                #h.evaluate()
                h.likelihood = h._compute_dataLogLikelihood(groupSize=-1, alpha=-1)
                h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
                #print(h.likelihood)

                hypothesis_candidates.append(h)

            # Pick the best
            h = max(hypothesis_candidates, key=lambda h: h.likelihood)
            #print(h.likelihood)
            
        best_dim_params = h.dimensional_parameters

        # Perform the update
        graph_function.dimensional_parameters = best_dim_params
        graph_function.kernels = graph_function._make_kernels(best_dim_params, graph_function.make_fn_decompositions())
        return h.likelihood

    def _optimize_hypotheses(self, hypothesis_graph):
        raise NotImplementedError
    def HypothesisType(self):
        raise NotImplementedError

class Context(object):
    def __init__(self, fully_optimize_lengthscales, p, M, max_group_size, sigma2, cycles, lengthscaleNumIter, opt_restart, param_exploration):
        # Decide if should optimize lengthscale
        self.fully_optimize_lengthscales = fully_optimize_lengthscales

        # Prior knowledge of getting an edge
        self.p = p

        # The number of clusters
        self.M = M
        # The maximum size of each cluster
        self.max_group_size = max_group_size

        # Stability parameter for data likelihood
        self.sigma2 = sigma2

        # Assume that there are cycles for prior
        self.cycles = cycles

        self.param_exploration = param_exploration

        self.gen_candidates = 20

        # Smallest possible value for ls
        self.ls_min_limit = 1e-4
        # This is essentially sqrt(0.1), we set the var limit to be under 0.1
        self.var_min_limit = 0.31622776601683794 
        self.var_max_limit = 1e5
        
        # Number of samples taken for the parameters
        self.lengthscaleNumIter = lengthscaleNumIter

        self.ln_limits = (1e-2, 1e5)
        self.var_limits = (0.31622776601683794, 1e5)

        self.opt_restart = opt_restart
    def kern_respect_bounds(self, param):
        ls, var = param
        _pp = (min(max(self.ln_limits[0], ls), self.ln_limits[1]), min(max(self.var_limits[0], var), self.var_limits[1]))
        return _pp
class Data(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.Y_r = self.Y.reshape(self.n, 1)

# There are different hypothesis representations
# ==============================================

# Base class to deal with the parameters and other misc stuff
class Hypothesis(object):
    def __init__(self, graph_function, context, data):

        self.ls_wrap = graph_function.ls_wrap
        self.scipy_opt = graph_function.scipy_opt

        self.dimensional_parameters = np.array(graph_function.dimensional_parameters)

        self.likelihood = 0.
        self.context = context
        self.data = data
        self._sk_kernel_class = graph_function._sk_kernel_class
        self._sk_kwargs = graph_function._sk_kwargs
        self.kernels = self._make_kernels(self.dimensional_parameters, graph_function.make_fn_decompositions())

    def has_cycle(self):
        return len(nx.cycle_basis(self.make_graph())) == 0
    def optimize_dimensional_parameters(self, dim_n_iter, groupSize = -1, alpha = 1):
        #return self.optimize_dimensional_old(dim_n_iter, groupSize, alpha)
        return self.optimize_dimensional_grads(dim_n_iter, groupSize, alpha)
        
    def optimize_dimensional_grads(self, dim_n_iter, groupSize = -1, alpha = 1):
        
        logging.info(cached_apply_X.cache_info())
        cached_apply_X.cache_clear()
        
        #_max_val = np.min(dim_n_iter * self.data.dim * self.context.param_n_iter,1000)
        _max_val = dim_n_iter * self.data.dim
        target_n = np.log10(_max_val)
        iters_decay = lambda t: 1e0 * np.exp((target_n-0)/1000.0*np.log(10)*t)
        iters = int(iters_decay(self.data.n))

        #print(self.dimensional_parameters)
        dim_ls, dim_var = map(np.array, zip(*self.dimensional_parameters))
        dim_param0 = np.concatenate( (dim_ls, dim_var) )
        # One lengthscale per dimension
        dim = len(self.dimensional_parameters)

        fn_decomps = self.make_fn_decompositions_sorted()
        
        # There are accuracy problems with lengthscale < 1e-2 
        # bounds = [(1e-2, 1e5)] * dim + [(1e-1, 1e5)] * dim
        bounds = [(1e-2, 1e5)] * dim + [(0.31622776601683794, 1e5)] * dim

        def dim_param_obj_min_f_df(dim_param):
            f, df = self.phi_f_df_cholesky_sk(dim_param[:dim], dim_param[dim:], self.kernels)
            return -f, -df

        # Cannot use the other 2 params as they differ algo to algo
        opt_x, _, _ = self.scipy_opt(dim_param_obj_min_f_df, approx_grad=False, x0=dim_param0, bounds=bounds, disp=0, maxfun=iters)
        
        dim_ls = opt_x[:dim]
        dim_var = opt_x[dim:]

        # Check if the dim param changed 
        _dimensional_parameters = np.array(list(map(np.array, zip(dim_ls, dim_var))))
        if np.allclose(_dimensional_parameters, self.dimensional_parameters):
            return self.likelihood

        self.dimensional_parameters = _dimensional_parameters
        self._update_kernels(dim_ls, dim_var, self.kernels)
        
        # Include Groupsize, need to recompute the likelihood
        updated_likelihood = self._compute_dataLogLikelihood(groupSize, alpha)
        #print(updated_likelihood, -opt_f)
        
        # Sanity check that it improved.
        if not np.isclose(updated_likelihood, self.likelihood):
            assert(updated_likelihood > self.likelihood)
        self.likelihood = updated_likelihood
        return self.likelihood
        #return self._compute_dataLogLikelihood(groupSize, alpha)
        
    def apply_X(self, kernel):
        return cached_apply_X(KernelWrap(kernel, self.data))
    def _compute_dataLogLikelihood(self, groupSize, alpha):

        #K = reduce(operator.add, [ kernel.K(self.data.X) for kernel in kernels ])
        K = reduce(np.add, map(self.apply_X, self.kernels.values()))

        #logp = self.phi_full(self.data.X, self.data.Y, K)
        logp = self.phi_cholesky(self.data.X, self.data.Y, K)
        #assert(np.allclose(logp, _logp))
        
        # This is reserved for non overlap
        if groupSize >= 0: # in the case of Gibbs Sampling
            logp += np.log(groupSize + alpha)

        return logp

    def phi_f_df_cholesky_sk(self, dim_ls, dim_var, kernels):

        # Noise inside
        noise_var = 0
        #likelihood = likelihoods.Gaussian(variance=noise_var)

        X = self.data.X
        y = self.data.Y

        Ky = np.zeros((X.shape[0],X.shape[0]))
        dK_dparams = []
        for var_order, k in kernels.items():
            
            _var_order = list(var_order)
            ls = self.ls_wrap(dim_ls[_var_order])
            _dim_var = dim_var[_var_order]
            variance = np.sqrt(sum( _dim_var**2 ))

            k._dim_var = _dim_var
            k.set_params(k1__constant_value=variance, k2__length_scale=ls)
            k.variance = variance
            k.lengthscale = ls

            K_part, dK_dparam = k(X[:,k.active_dims], eval_gradient=True)
            Ky += K_part
            dK_dparams.append(dK_dparam)


        np.fill_diagonal(Ky, Ky.diagonal() + self.context.sigma2)

        LW = np.linalg.cholesky(Ky)
        c = np.linalg.inv(LW)
        Wi = np.dot(c.T,c)
        
        W_logdet = 2 * np.sum(np.log(LW.diagonal()))

        alpha = scipy.linalg.cho_solve((LW, True), y)
        n = X.shape[0]
        log_2_pi = np.log(2*np.pi)*n/2.0

        log_marginal = -0.5*(W_logdet + np.dot(y, alpha))

        dL_dK = 0.5 * (np.einsum('i,k->ik', alpha, alpha, dtype=np.float64) - Wi)
        
        grad_dim = np.zeros(self.data.dim*2)
        grad_dim_ls = grad_dim[:self.data.dim]
        grad_dim_var = grad_dim[self.data.dim:]

        for k, dK_dparam in zip(kernels.values(), dK_dparams):

            dL_dparam = np.einsum('ij,jik->k', dL_dK, dK_dparam)

            dL_dsigma = dL_dparam[0] * k._dim_var / k.variance ** 2
            grad_dim_var[k.active_dims] += dL_dsigma

            dL_dls = dL_dparam[1:] / k.lengthscale
            grad_dim_ls[k.active_dims] += dL_dls

        # Fix Strange problems by numerical instability
        if log_marginal > log_2_pi:
            log_marginal = log_2_pi
            # Cannot afford to increase the L anymore, so make the gradient really small and in the opposite dir
            grad_dim = grad_dim * -1e-8
        
        return log_marginal, grad_dim

    # Also known as logp
    def phi_cholesky(self, X, y, Ky):

        # Ky = K + sigma^2 I
        np.fill_diagonal(Ky, Ky.diagonal() + self.context.sigma2)

        # We decompose the matrix for constant use
        LW = np.linalg.cholesky(Ky)

        # We skip the computation of the inverse and solve for alpha
        # alpha = inv(Ky) * y
        alpha = scipy.linalg.cho_solve((LW, True), y)

        # Determinant using choleskey
        # W_logdet = log|Ky|
        # log(det(Ky)) = log(det(LW) * det(LW_t)) = 2 log(det(L)) = 2 log(Product(diag(L))) = 2 sum( log (dia(L)) )
        W_logdet = 2 * np.sum(np.log(LW.diagonal()))
        
        # term3 - ignore to save computation
        n = X.shape[0]
        log_2_pi = np.log(2*np.pi)*n/2.0

        # finally compute term1 + term2 w/o term3
        # We note that term3 is not useful as its the same throughout.
        log_marginal = -0.5*(np.dot(y, alpha) + W_logdet) 
        #assert(log_marginal <= log_2_pi)
        #log_marginal = -0.5*(np.linalg.dot(y, alpha) + W_logdet + log_2_pi)
        
        # Fix Strange problems by numerical instability
        if log_marginal > log_2_pi:
            log_marginal = log_2_pi

        return log_marginal 
    # Decide if optimizing lengthscale is needed
    def evaluate(self, groupSize = -1, alpha = 1):
        if self.context.fully_optimize_lengthscales:
            self.likelihood = self.optimize_dimensional_parameters(self.context.lengthscaleNumIter, groupSize, alpha)
        else:
            self.likelihood = self._compute_dataLogLikelihood(groupSize, alpha)

        return self.likelihood
    def dimension(self):
        return len(self.dimensional_parameters)
    def make_fn_decompositions_sorted(self):
        raise NotImplementedError
    def make_graph(self):
        raise NotImplementedError
    def clone(self):
        h = type(self).__new__(self.__class__)
        h.likelihood = 0.
        h.dimensional_parameters = self.dimensional_parameters.copy()
        h.context = self.context
        h.data = self.data
        h.kernels = None
        h._sk_kernel_class = self._sk_kernel_class
        h._sk_kwargs = self._sk_kwargs
        h.ls_wrap = self.ls_wrap
        h.scipy_opt = self.scipy_opt
        return h
    def _make_kernels(self, dimensional_parameters, fn_decompositions, prev_kernels={}):

        dim_ls, dim_var = map(np.array, zip(*dimensional_parameters))
        nActiveVar = sum(map(len, fn_decompositions))
        kernels = {}
        for var_order in fn_decompositions:

            if var_order in prev_kernels:
                kernels[var_order] = prev_kernels[var_order]
                continue

            d = len(var_order)
            # Prevent the values from going heywire
            #ls = normalize(np.clip(dimensional_parameters[var_order], 1e-03, 1))
            _var_order = list(var_order)
            ls = self.ls_wrap(dim_ls[_var_order])
            _dim_var = dim_var[_var_order]
            variance = np.sqrt(sum( _dim_var**2 ))
            
            #var = float(d) / nActiveVar
            logging.debug("Fn={}, ls={}, variance={}".format(var_order, ls, variance))
            kernel = variance * self._sk_kernel_class(ls, **self._sk_kwargs)
            kernel._dim_var = _dim_var
            kernel.active_dims = _var_order
            kernel.variance = variance
            kernel.lengthscale = ls

            kernels[var_order] = kernel
        return kernels

    def _update_kernels(self, dim_ls, dim_var, kernels):

        for var_order, kernel in kernels.items():
            
            _var_order = list(var_order)
            ls = self.ls_wrap(dim_ls[_var_order])
            _dim_var = dim_var[_var_order]
            variance = np.sqrt(sum( _dim_var**2 ))

            kernel._dim_var = _dim_var
            kernel.set_params(k1__constant_value=variance, k2__length_scale=ls)
            kernel.variance = variance
            kernel.lengthscale = ls

        return kernels

# We operate everything based an the adjacency matrix, as underlying structure is graph
class HypothesisGraph(Hypothesis):
    def __init__(self, graph_function, context, data):
        self.Z = nx.to_numpy_matrix(graph_function.graph, dtype=bool)
        super().__init__(graph_function, context, data)
    # Maybe can be in base?
    def make_fn_decompositions_sorted(self):
        return [ tuple(sorted(ea_decomp)) for ea_decomp in nx.find_cliques(self.make_graph()) ]
    def flip_edge(self, i, j):
        Z = self.Z.copy()
        Z[i,j] = not Z[i,j]
        Z[j,i] = Z[i,j]

        # Update for edge on likihood
        # WHY?? TODO
        # Update edge perturbation, TODO merge?
        is_edge_set = self.Z[i,j] == 1
        '''
        if is_edge_set:
            likelihood = np.log(self.context.p)
        else:
            likelihood = np.log(1-self.context.p)
        '''

        h = self.clone()
        #h.likelihood = likelihood
        h.Z = Z
        h.kernels = h._make_kernels(h.dimensional_parameters, h.make_fn_decompositions_sorted(), self.kernels)
        return h, is_edge_set
    def mutate_edge(self, i_del, j_del, i_add, j_add):
        Z = self.Z.copy()
        Z[i_add,j_add] = True
        Z[j_add,i_add] = True
        Z[i_del,j_del] = False
        Z[j_del,i_del] = False

        # Update for edge on likihood
        # WHY?? TODO
        # Update edge perturbation, TODO merge?
        #likelihood = np.log(self.context.p)

        h = self.clone()
        #h.likelihood = likelihood
        h.Z = Z
        h.kernels = h._make_kernels(h.dimensional_parameters, h.make_fn_decompositions_sorted(), self.kernels)
        return h
    def make_graph(self):
        return nx.from_numpy_matrix(self.Z)

# We operate everything based on adj list
class HypothesisNonGraph(Hypothesis):
    def __init__(self, graph_function, context, data):
        self.z = self.getZFromGraph(graph_function.graph)
        super().__init__(graph_function, context, data)
    def getZFromGraph(self, G, M=0):
        z = np.zeros(len(G.nodes()))
        cliques = nx.find_cliques(G)
        group = 0
        for c in cliques:
            for n in c:
                z[n] = group
            group += 1
        '''
        if M > 0:
            assert(group <= M) # The number of groups must not exceed M
        '''
        return z
    def make_fn_decompositions_sorted(self):
        M = int(max(self.z)+1)
        decomp = []
        #values = set(np.array(z).flatten())
        for m in range(M):
            A = []
            for j, z_j in enumerate(self.z):
                #assert(z_j < M) # the constraint on the number of groups must be fullfilled
                if z_j == m:
                    A.append(j)
            if len(A) > 0:
                decomp.append(tuple(sorted(A)))
        
        # Fixed missing nodes
        activated_set = reduce(set.union, map(set, map(list, decomp)))
        leftover = set(range(self.data.dim)) - activated_set
        
        decomp += [ tuple([i]) for i in leftover ]
        
        return decomp
    def make_graph(self):
        decomp = self.make_fn_decompositions_sorted()
        return self.getGraphFromDecomposition(decomp)
    def getGroupSize(self, decomp, j):
        for d in decomp:
            if j in d:
                return len(d)
        return 0
    def getGraphFromDecomposition(self, decomp):
        graph = nx.Graph()
        edges = []
        single_nodes = []
        for v in decomp:
            if len(v) > 1:
                all_pairs = list(itertools.combinations(v, 2))
                for pair in all_pairs:
                    edges.append(pair)
            elif len(v) == 1:
                single_nodes.append(v[0])
        graph.add_edges_from(edges)
        graph.add_nodes_from(single_nodes)
        return graph
    def update_group(self, j, m, omega=0.):
        z = self.z.copy()
        z[j] = m

        h = self.clone()
        h.likelihood = omega
        h.z = z
        h.kernels = h._make_kernels(h.dimensional_parameters, h.make_fn_decompositions_sorted(), self.kernels)
        return h
    def maxGroupSize(self, decomp):
        m = 0
        for d in decomp:
            s = len(d)
            if s > m:
                m = s
        return m
        
class GraphNonOverlap(GraphOptimizer):
    def __init__(self, cycles, max_group_size, **kwargs):
    
        # GraphNonOverlap
        # No cycles constraint, change group size to 2
        if not cycles:
            # Constraints to group size 2
            logging.info("Overriding max_group_size to 2 as no cycles are reported for GraphNonOverlap.")
            max_group_size = 2

        GraphOptimizer.__init__(self, cycles=cycles, max_group_size=max_group_size, **kwargs)
    def _optimize_hypotheses(self, h_0):

        dim = h_0.dimension()
        fully_optimize_lengthscales = self.context.fully_optimize_lengthscales

        best_h = h_0
        h_prev = h_0
        count_i=0
        # num_iter = int(self.graphSamplingNumIter / (dim*self.context.M)) + 1
        while True:
            h = h_prev
            dimensional_parameter_new = None
            for j in np.random.permutation(dim): # Sample z_j from p(z_j = m | z_-j, D) \proto exp(phi_m) using Gumbel's trick
                omega = np.random.gumbel(0.0, 1.0, dim)

                decomp = h.make_fn_decompositions_sorted()
                size = h.getGroupSize(decomp, j)

                # Partial evaluation of m
                # TODO Replace this with a loop_best 
                z_j_new = -1
                best_val = -np.inf
                for m in range(self.context.M): # select z_j = argmax_{i <= M} phi_i + omega_i

                    h = h.update_group(j, m, omega[m])
                    decomp = h.make_fn_decompositions_sorted()

                    # Nth to consider if its maxed out alr.
                    if h.maxGroupSize(decomp) > self.context.max_group_size:
                        continue

                    h.evaluate(groupSize=size)
                    count_i += 1

                    # maxGroupSize
                    if h.likelihood > best_val:
                        best_val = h.likelihood
                        z_j_new = m
                        if fully_optimize_lengthscales:
                            dimensional_parameter_new = np.copy(h.dimensional_parameter)
                    
                    if count_i >= self.graphSamplingNumIter:
                        break
                
                h = h.update_group(j, z_j_new)
                if fully_optimize_lengthscales:
                    h.dimensional_parameters = dimensional_parameter_new

                if count_i >= self.graphSamplingNumIter:
                    break

            # without group bias
            h.evaluate()
            if h.likelihood > best_h.likelihood:
                best_h = h
                if fully_optimize_lengthscales:
                    best_dimensional_parameter = dimensional_parameter_new.copy()      
            h_prev = h

            if count_i >= self.graphSamplingNumIter:
                break

        if not fully_optimize_lengthscales:
            best_h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
        else:
            best_h.dimensional_parameters = best_dimensional_parameter

        return best_h

    def HypothesisType(self):
        return HypothesisNonGraph

class GraphOverlap(GraphOptimizer):
    def _optimize_hypotheses(self, h_0):
        logging.info("Running GraphOverlap")
        dim = h_0.dimension()
        h_prev = h_0
        h = h_0
        best_h = h_0

        hypotheses_set = set([ h_0 ])
        sampled_hypotheses_set = set([ h_0 ])

        all_edges = [ (i, j) for i in np.random.permutation(dim) for j in np.random.permutation(i) ]

        while len(hypotheses_set) < self.graphSamplingNumIter:
            np.random.shuffle(all_edges)
            for i, j in all_edges:
                h_prev = h
                h, is_edge_set = h.flip_edge(i, j)

                # Check if the graph satisfy the prior cycle condition
                if not (not self.context.cycles and is_edge_set) or h.has_cycle():
                    # Check if its a new hypothesis
                    if not h in hypotheses_set:
                        h.evaluate()
                        hypotheses_set.add(h)

                    # Check if the current hypothesis is better then the prev graph
                    if h.likelihood < h_prev.likelihood :
                        # Worse than previous graph -> Switch the edge back again
                        h = h_prev
                else:
                    # Illegal operation, reset back
                    h = h_prev

                if len(hypotheses_set) >= self.graphSamplingNumIter:
                    break

            if h.likelihood > best_h.likelihood:
                best_h = h

            # Choose as next sample the graph with highest likelihood which has not
            # yet been selected as sample.
            # Continue to use h if we know it is still the best
            if not h in sampled_hypotheses_set: 
                # h has been updated, not been sampled from before
                # We choose the next best
                unsampled_hypotheses_set = hypotheses_set - sampled_hypotheses_set
                # this sample has already been selected, find a new one that is probable
                # maximum probability
                h = max(unsampled_hypotheses_set, key=lambda h:h.likelihood)
                sampled_hypotheses_set.add(h_prev)

        # Did not perform full optimization, so we optimze at the end
        if not self.context.fully_optimize_lengthscales:
            best_h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)

        #assert(len(hypotheses_set) == self.graphSamplingNumIter)
        return best_h
    def HypothesisType(self):
        return HypothesisGraph

class Tree(GraphOptimizer):
    def HypothesisType(self):
        return HypothesisGraph
    def _optimize_hypotheses(self, h_0):

        logging.info("Running Tree")
        dim = h_0.dimension()
        h_prev = h_0
        h = h_0
        best_h = h_0

        hypotheses_set = set([ h_0 ])
        sampled_hypotheses_set = set([ h_0 ])
        
        all_edges = [ (i, j) for i in np.random.permutation(dim) for j in np.random.permutation(i) ]

        while len(hypotheses_set) < self.graphSamplingNumIter:

            if len(h.make_graph().edges()) < dim - 1:
                
                edges = h.make_graph().edges()
                disjoint_set = DisjointSet()
                for i, j in edges:
                    disjoint_set.union(i,j)

                np.random.shuffle(all_edges)
                #from tqdm import tqdm
                #for i, j in tqdm(all_edges):
                for i, j in all_edges:
                    # Checks for the same parent, which will check for cycle.
                    # TODO DEBUG, to make sure cycles does not exist
                    #print(i, j, parent)
                    if not disjoint_set.connected(i, j):

                        h_prev = h
                        h, _ = h.flip_edge(i, j)

                        # Check if its a new hypothesis
                        if not h in hypotheses_set:
                            h.evaluate()
                            hypotheses_set.add(h)

                        # Check if the current hypothesis is better then the prev graph
                        if h.likelihood < h_prev.likelihood:
                            # Worse than previous graph -> Switch the edge back again
                            h = h_prev
                        else:
                            disjoint_set.union(i, j)
                    
                    if len(hypotheses_set) >= self.graphSamplingNumIter:
                        break

                    if len(h.make_graph().edges()) >= dim - 1:
                        break
                #assert(len(edges) <= len(h.make_graph().edges()))
            else:
                graph_orig = h.make_graph()
                edges = list(graph_orig.edges())
                h_prev = h
                h_orig = h
                # we generate candidates from the original hypothesis
                for i in range(self.context.gen_candidates):
                    graph_copy = graph_orig.copy()
                    # The edge to remove
                    i_del, j_del = edges[np.random.choice(len(edges))]
                    graph_copy.remove_edge(i_del, j_del)
                    comp_1, comp_2 = list(nx.descendants(graph_copy, i_del)) + [i_del], list(nx.descendants(graph_copy, j_del)) + [j_del]
                    
                    candidate_edges = set(product(comp_1, comp_2))
                    # We do not want to add it back again, its lame...
                    # Unless its the case where there is nothing else
                    if len(candidate_edges) > 1:
                        candidate_edges.remove((i_del, j_del))
                    candidate_edges = list(candidate_edges)

                    i_add, j_add = candidate_edges[np.random.choice(len(candidate_edges))]
                    #for i_add, j_add in candidate_edges:
                    
                    h_prev = h
                    h = h_orig.mutate_edge(i_del, j_del, i_add, j_add)

                    # Check if its a new hypothesis
                    if not h in hypotheses_set:
                        h.evaluate()
                        hypotheses_set.add(h)

                    # Check if the current hypothesis is better then the prev graph
                    if h.likelihood < h_prev.likelihood:
                        # Worse than previous graph -> Switch the edge back again
                        # H is only replaced if h is actually better
                        h = h_prev

                    if len(hypotheses_set) >= self.graphSamplingNumIter:
                        break
                #assert(len(edges) == len(h.make_graph().edges()))

            if h.likelihood > best_h.likelihood:
                best_h = h

            # Choose as next sample the graph with highest likelihood which has not
            # yet been selected as sample.
            # Continue to use h if we know it is still the best
            if not h in sampled_hypotheses_set: 
                # h has been updated, not been sampled from before
                # We choose the next best
                unsampled_hypotheses_set = hypotheses_set - sampled_hypotheses_set
                # this sample has already been selected, find a new one that is probable
                # maximum probability
                h = max(unsampled_hypotheses_set, key=lambda h:h.likelihood)
                sampled_hypotheses_set.add(h_prev)

        # Did not perform full optimization, so we optimze at the end
        if not self.context.fully_optimize_lengthscales:
            best_h.optimize_dimensional_parameters(self.context.lengthscaleNumIter)
        
        #print(best_h.dimensional_parameters)
        #assert(len(hypotheses_set) == self.graphSamplingNumIter)
        
        return best_h


class ParameterOnlyOptimizer(GraphOptimizer):
    def HypothesisType(self):
        return HypothesisGraph

#from profilehooks import profile
import operator
from functools import reduce

class GraphFunction(object):
    def __init__(self, graph, initial_kernel_params):
        self.graph = graph
        self.initial_kernel_params = initial_kernel_params
        self.dimensional_parameters = [ (initial_kernel_params['lengthscale'], initial_kernel_params['variance']) for i in range(self.dimension()) ]
        # dangerous but ok...
        # TODO

        # Set kernels, if not set then use defaults
        if 'gpy_kernel' in initial_kernel_params:
            self._gpy_kernel_class = self.locate(initial_kernel_params['gpy_kernel'])
        else:
            self._gpy_kernel_class = GPy.kern.RBF

        self._sk_kwargs = {}
        if 'sk_kernel' in initial_kernel_params:
            self._sk_kernel_class = self.locate(initial_kernel_params['sk_kernel'])
            if 'sk_kwargs' in initial_kernel_params:
                self._sk_kwargs = initial_kernel_params['sk_kwargs']
        else:
            self._sk_kernel_class = sklearn.gaussian_process.kernels.RBF

        # For future use, to expand the building of reusable kernels and creating only differences.
        # TODO

        # Decide for ARD 
        # TODO hack
        self.is_ard = True
        if 'ard' in initial_kernel_params:
            self.is_ard = initial_kernel_params['ard']
        
        if self.is_ard:
            self.ls_wrap = lambda ord_dim_ls: ord_dim_ls
        else:
            self.ls_wrap = lambda ord_dim_ls: sum(ord_dim_ls)

        # l_bfgs is now the default.
        self.scipy_opt = scipy.optimize.fmin_tnc
        if 'scipy_opt' in initial_kernel_params:
            self.scipy_opt = self.locate(initial_kernel_params['scipy_opt'])

        fn_decompositions = self.make_fn_decompositions()
        self.kernels = self._make_kernels(self.dimensional_parameters, fn_decompositions)
    def dimension(self):
        return self.graph.number_of_nodes()
    def make_decomposition(self, model):
        fn_decompositions = self.make_fn_decompositions()
        cfns = self.make_cfns(self.kernels, model)
        return (fn_decompositions, GPy.kern.Add(self.kernels.values()), cfns)
    def make_fn_decompositions(self):
        return [ tuple(sorted(ea_decomp)) for ea_decomp in nx.find_cliques(self.graph) ]
    def locate(self, path):
        # Dynamically load the class
        (modulename, classname) = path.rsplit('.', 1)
        m = __import__(modulename, globals(), locals(), [classname])
        if not hasattr(m, classname):
            raise ImportError(f'Could not locate "{path}".')
        return getattr(m, classname)
    def _make_kernels(self, dimensional_parameters, fn_decompositions, prev_kernels={}):
        dim_ls, dim_var = map(np.array, zip(*dimensional_parameters))
        nActiveVar = sum(map(len, fn_decompositions))
        kernels = {}
        for var_order in fn_decompositions:

            if var_order in prev_kernels:
                kernels[var_order] = prev_kernels[var_order]
                continue

            d = len(var_order)
            # Prevent the values from going heywire
            #ls = normalize(np.clip(dimensional_parameters[var_order], 1e-03, 1))
            _var_order = list(var_order)
            ls = self.ls_wrap(dim_ls[_var_order])
            _dim_var = dim_var[_var_order]
            variance = np.sqrt(sum( _dim_var**2 ))
            
            #var = float(d) / nActiveVar
            logging.debug("Fn={}, ls={}, variance={}".format(var_order, ls, variance))
            kernel = self._gpy_kernel_class(input_dim=d, lengthscale=ls, variance=variance, active_dims=var_order, ARD=self.is_ard, name="_"+"_".join(map(str,var_order)) )
            kernel._dim_var = _dim_var
            kernel.fix()
            kernels[var_order] = kernel
        return kernels

    def _update_kernels(self, dim_ls, dim_var, kernels):

        for var_order, kernel in kernels.items():
            
            _var_order = list(var_order)
            new_ls = self.ls_wrap(dim_ls[_var_order])
            _dim_var = dim_var[_var_order]

            kernel.lengthscale = new_ls
            kernel.variance = np.sqrt(sum( _dim_var**2 ))
            kernel._dim_var = _dim_var

        return kernels

    def make_cfns(self, kernels, model):
        fn_decomp_lookup = {}
        for var, kernel in kernels.items():
            acqu = MyAcquisitionLCB(model, kernel, var)
            fn_decomp_lookup[var] = acqu
        return ComponentFunction(fn_decomp_lookup)

    def copy(self):
        return GraphFunction(self.graph.copy(), self.initial_kernel_params)

    def copy_randomised(self, size_of_random_graph):
        return GraphFunction(get_random_graph(self.dimension(), max(1, int(self.dimension()*size_of_random_graph))), self.initial_kernel_params)

class OptimalGraphFunction(GraphFunction):
    def make_cfns(self, kernels, model):
        fn_decomp_lookup = {}
        for var, kernel in kernels.items():
            acqu = MyAcquisitionLCB(model, kernel, var)
            fn_decomp_lookup[var] = acqu
        return SyntheticComponentFunction(self.graph, fn_decomp_lookup)

    def make_fn_decompositions(self):
        return [ tuple(sorted([v])) for v in nx.isolates(self.graph)] + [ tuple(sorted(e)) for e in self.graph.edges() ]

    def _make_kernels(self, dimensional_parameters, fn_decompositions, prev_kernels={}):
        ls, var = dimensional_parameters[0]
        kernels = {}
        for var_order in fn_decompositions:
            logging.info("Fn={}, ls={}, variance={}".format(var_order, ls, var))
            kernel = GPy.kern.RBF(input_dim=len(var_order), lengthscale=ls, variance=var, active_dims=var_order, name="_"+"_".join(map(str,var_order)))
            kernel.fix()
            kernels[var_order] = kernel
        return kernels