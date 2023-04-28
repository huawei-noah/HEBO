from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
import numpy as np
import networkx as nx
import itertools
import logging
import math
from itertools import product

from graph_utils import make_chordal, build_clique_graph
from mlflow_logging import MlflowLogger
from functools import reduce

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Fast join
sorted_tuple = lambda x: tuple(sorted(x))

class MPAcquisitionOptimizer(AcquisitionOptimizer):
    """
    Class for optimizing the acquisition function in the considering that is of the form : f(x1,x2,x3,x4) = f1(x1,x2) + f2(x3) + f3(x4)
    """
    def __init__(self, domain, graph_function, X, mlflow_logging, max_eval, acq_opt_restarts=1):
        super(MPAcquisitionOptimizer, self).__init__(domain)
        self.domain = domain
        self.max_eval = max_eval
        self.graph_function = graph_function
        self.acq_opt_restarts = acq_opt_restarts
        self.dimension = graph_function.graph.number_of_nodes()

        self.edges_set = None
        self.init_graph_helpers()
        
        # ==========================================
        self.evaluated_points = []
        for x in X: # we should have already picked at least one point randomly
                self.evaluated_points.append(x) 
        # ==========================================
        
        self.mlflow_logging = mlflow_logging
        self._total_cost = 0

    def _optimize(self, f):

        inital_total_cost = self._total_cost

        evals = []
        for i in range(self.acq_opt_restarts):
            self._total_cost = inital_total_cost
            x_best, fmin = self._optimize_helper(f)
            total_cost = self._total_cost
            evals.append((fmin, x_best, total_cost))

        fmin, x_best, total_cost = min(evals, key=lambda x: x[0])
        self._total_cost = total_cost

        return x_best, fmin

    def _optimize_helper(self, f):
        # Optimizes each acquisition function on all disconnected trees
        x_best = self.domain.none_value()
        fmin = 0
        #disconnectedComp = list(nx.components.connected.connected_component_subgraphs(self.clique_tree))
        disconnectedComp = [ self.clique_tree.subgraph(c) for c in nx.connected_components(self.clique_tree) ]
        np.random.shuffle(disconnectedComp)

        for clique_tree in disconnectedComp: # loop on all disconnected trees
            fmin_tree, x_best_tree = self.optimize_connected_tree(clique_tree)
            x_best[x_best_tree != -1] = x_best_tree[x_best_tree != -1]
            
            fmin += fmin_tree

        return x_best, fmin

    def optimize(self, f=None, df=None, f_df=None):
   
        self._total_cost = 0
        self._f = f

        self.init_graph_helpers()

        #Original code 
        #=============
        # Make the domain
        self._domains = self.domain.get_opt_domain()['domain']
        if self.max_eval > 0:
            #logging.info("T_Max used")
            self._domains = self.make_small_domains(self._domains, self.max_eval)
            x_best, fmin = self._optimize(f)
        else:
            if self.max_eval < -1:
                #logging.info("Zooming used")
                target_grid = -(self.max_eval)
                subdiv_count = 0
                while True:
                    is_stop = True
                    # Now we do the partitioning
                    domain_chunks = []
                    for domain in self._domains:
                        if len(domain) > target_grid:
                            # The case where you can chop it up
                            is_stop = False
                            chunk_len = math.ceil(len(domain) / float(target_grid))
                            domain_chunks.append(list(chunks(domain, chunk_len)))
                        else:
                            domain_chunks.append([ np.array([d]) for d in domain ])
                    self._domains = [ [ np.random.choice(d, 1)[0] for d in domain_chunk] for domain_chunk in domain_chunks ]
                    x_best, fmin = self._optimize(f)
                    self._domains = []
                    
                    if is_stop:
                        break

                    # Reconstruct the 
                    temp_no = 0
                    for domain, x in zip(domain_chunks, x_best):
                        temp_no += 1
                        for d in domain:
                            if x in d:
                                self._domains.append(d)
                    subdiv_count += 1

                #self._domains = self.make_small_domains(self._domains, self.max_eval)
            else:
                #logging.info("Through optimization")
                assert(self.max_eval != 0)
                x_best, fmin = self._optimize(f)
        
        # Domain sensitive
        # ==========================================
        while list(x_best) in [list(item) for item in self.evaluated_points]:
            logging.fatal("Selected point already evaluated, performing perturbation")
            var = random.randint(0, len(x_best)-1)
            up = random.randint(0,1)
            i = np.where(self._domains[var] == x_best[var])[0]
            if i == 0 or (up == 1 and i < len(self._domains[var])-1):
                x_best[var] = self._domains[var][i+1]
            else:
                x_best[var] = self._domains[var][i-1]
        # ==========================================
        
        x_best = np.array([x_best])
        if self.mlflow_logging != None:
            self.mlflow_logging.log_cost(self._total_cost)

        # ============================================================
        # Basic checks
        assert(np.isclose(f(x_best), fmin)), f"{f(x_best)}!={fmin}"
        '''
        rranges = np.array([ self._domains[k] for k in range(len(self._domains))])
        prranges = np.array([ len(r) for r in rranges ])
        print(np.prod(prranges))
        print("Decomposition", f.keys())
        print("Domain", rranges)

        if self.max_eval != -1 and np.prod(prranges) < 40000 and False:
            print("Assert Checking for optimizer")
            # Additional check to ensure that the domain is respected
            for xi, di in zip(x_best[0], rranges):
                assert(xi in di)

            # We now brute force on the reduced domain
            f_min_bf = np.inf
            x_bruteforce = None
            for x in product(*rranges):
                x = np.array(x)
                f_min_curr = f(x)
                if f_min_curr < f_min_bf:
                    f_min_bf = f_min_curr
                    x_bruteforce = x     
            assert(np.all(np.isclose(x_bruteforce, x_best[0])))   
        # ============================================================
        '''

        return x_best, fmin, self._total_cost

    # Note that the reduction must be ordered
    # TODO 
    def get_subspace(self, subspace_var_order):
        # subspace_var_order must be list
        subdomains = [self._domains[i] for i in subspace_var_order]
        return itertools.product(*subdomains)

    def optimize_connected_tree(self, tree):
        self._computed_cliques_G = set()
        
        # We unroll the first call for message_passing, by starting with the max degree node
        root = max(tree.degree(), key=lambda x:x[1])[0]
        children = list(self.clique_tree.neighbors(root))        
        
        input_messages = self.broadcast_mp_children(root, children)

        intersection = root - frozenset().union(*children)
        marginal = root - intersection

        # Unpack
        marginal_var_order = sorted(marginal)

        if len(intersection) == 0:
            # Special case where we need to optimize directly for the outstanding variables
            filtered_cliques_G = self.filter_cliques_G(root)
            
            new_X = np.full((1, self.domain.dimension), -1, dtype=np.float)
            
            fmin_overall, x_best_overall = self.optimize_discrete(new_X, children, marginal_var_order, input_messages, root, filtered_cliques_G)
            fmin_overall = fmin_overall[0]
            x_best_overall = x_best_overall[0]
        else:

            # We ignore the message, as we are at the root
            fmin_overall, x_best_overall, _ = self.compute_message(intersection, children, marginal_var_order, input_messages, root)

        # Returns the element for which the value function is minimal (the min function selects with respect to the first value of the list)        
        return fmin_overall, x_best_overall

    def broadcast_mp_children(self, root, children):
        input_messages = []
        for child in children:
            child_messages = self.message_passing(child, root)
            input_messages.append(child_messages)
        return input_messages

    def compute_message(self, intersection, children, marginal_var_order, input_messages, root):

        # Filter cliques for computational efficieny
        filtered_cliques_G = self.filter_cliques_G(root)
    
        # Unpack
        intersection_var_order = sorted(intersection)
        #logging.info(f'intersection_var_order {intersection_var_order}')
        subspace_x = np.array(list(self.get_subspace(intersection_var_order)))
        new_X = np.full((subspace_x.shape[0], self.domain.dimension), -1, dtype=np.float)
        new_X[:,intersection_var_order] = subspace_x

        fmin_curr, x_best_curr = self.optimize_discrete(new_X, children, marginal_var_order, input_messages, root, filtered_cliques_G)
        
        # Compose the message
        output_message = { tuple(sub_x) : (fmin_ea, x_best_ea) for sub_x, fmin_ea, x_best_ea in zip(subspace_x, fmin_curr, x_best_curr) }

        _fmin_i = np.argmin(fmin_curr)
        fmin_overall = fmin_curr[_fmin_i]
        x_best_overall = np.copy(x_best_curr[_fmin_i])

        self._computed_cliques_G.update(filtered_cliques_G)
        return fmin_overall, x_best_overall, output_message

    def filter_cliques_G(self, root):
        subgraph_G = self.graph.subgraph(root)
        cliques_G = nx.find_cliques(subgraph_G)
        # Slightly more efficient?
        # [ s for s in map(frozenset, cliques_G) if not s in self._computed_cliques_G ]
        # return set(map(tuple, map(sorted, cliques_G))) - self._computed_cliques_G
        return set(map(sorted_tuple, cliques_G)) - self._computed_cliques_G

    def message_passing(self, root, parent):
        # For all values of the variables in intersection(root, parent), returns
        # the minimum value of the acquisition function for the variables in root-parent,
        # and the associated x for the relevant variables (0 for the others).
        
        children = list(self.clique_tree.neighbors(root))
        children.remove(parent) # note that neighbours would include parents
        
        input_messages = self.broadcast_mp_children(root, children)

        intersection = root.intersection(parent)        # Notice that len(intersection) != 0
        marginal = root - intersection
        
        # Unpack        
        marginal_var_order = sorted(marginal)

        # Quantities to return when optimizing the root
        fmin_overall, x_best_overall, output_message = self.compute_message(intersection, children, marginal_var_order, input_messages, root)

        return output_message

    def optimize_discrete(self, new_X, children, marginal_var_order, input_messages, root, filtered_cliques_G):
        fmin_X = np.full(new_X.shape[0], np.inf, dtype=np.float)
        # fmin_X = []
        X_best = np.full((new_X.shape[0], self.domain.dimension), np.inf, dtype=np.float)
        #x_best = None
        
        # Prepare the intersection_child_var_order
        # Unpack
        intersection_child_var_orders = [ sorted(root.intersection(c)) for c in children ]

        # Unpack
        for x_marginal in self.get_subspace(marginal_var_order):
            
            new_X[:,marginal_var_order] = x_marginal

            f_X_marginal = self.load_messages(new_X, input_messages, intersection_child_var_orders) # also modifies new_x
            
            # 2- Compute overall function for this node and optimise it
            f_X_marginal += self.compute_node_function(filtered_cliques_G, new_X)
            
            self._total_cost += new_X.shape[0]

            fmin_update_indices = fmin_X > f_X_marginal
            X_best[fmin_update_indices] = new_X[fmin_update_indices]
            fmin_X[fmin_update_indices] = f_X_marginal[fmin_update_indices]

        return fmin_X, X_best

    def load_messages(self, new_X, input_messages, intersection_child_var_orders):
        # modify new_x with messages received from 'children' and returns the sum of all function values of the children nodes
        f_x_marginal = np.zeros(new_X.shape[0])
        for m, intersection_child_var_order in zip(input_messages, intersection_child_var_orders):

            marginal_x_child = new_X[:,intersection_child_var_order]
            fmin_child, best_x_child = map(np.array, zip(*map(lambda sub_x_child: m[tuple(sub_x_child)], marginal_x_child)))
            
            # Update the marginals and x
            # replace 0 values in new_x with ones in best_x_child
            f_x_marginal += fmin_child
            new_X[best_x_child != -1] = best_x_child[best_x_child != -1]

        return f_x_marginal

    def compute_node_function(self, filtered_cliques_G, new_X):
        fval = np.zeros(new_X.shape[0])
        # For all cliques c of the subset of the triangulated clique 
        # if clique_G in self._f:
        for clique_G in filtered_cliques_G.intersection(self._f.keys()):
            fval += self._f[clique_G](new_X).reshape(-1)
            # Cases that will be excluded here, 
            # 1. cliques that does not have any computable edges
            # 2. cliques that are partially cut off
            # 3. cliques that are previous computed
            #print(clique)
            #assert(False)
        return fval
    def make_small_domains(self, domains, max_eval):
        small_domains = {}
        #disconnectedComp = list(nx.components.connected.connected_component_subgraphs(self.triangulated_graph))
        disconnectedComp = [ self.triangulated_graph.subgraph(c) for c in nx.connected_components(self.triangulated_graph) ]
        nDiscComp = len(disconnectedComp)
        max_eval += self.overflow(domains, max_eval)
        for subgraph in disconnectedComp:
            N_subgraph = max_eval * subgraph.number_of_nodes() / self.triangulated_graph.number_of_nodes() # number of evaluations for this component
            n_clique_var = 0
            for clique in nx.find_cliques(subgraph):
                n_clique_var += len(clique)
            for var in subgraph.nodes():
                clique_size = nx.node_clique_number(subgraph, var)
                N_clique = clique_size * N_subgraph / n_clique_var # number of evaluation for this clique
                N_var = max(int(round(N_clique**(1.0/clique_size))), 2)
                small_domains[var] = self.choose_from(domains[var], N_var)
        return small_domains
    def choose_from(self, domain, N_var):
        # randomly selects N_var DIFFERENT points
        if N_var >= len(domain):
            return np.array(domain)
        # Cannot be replaced, items drawn must be unique
        return np.random.choice(list(domain), N_var, replace=False)
    def overflow(self, domains, max_eval):
        # only consider overflow for size 1 groups
        overflow = 0
        for n in self.triangulated_graph.nodes():
            # TODO change to degree?
            # self.triangulated_graph.degree(n)
            if len(list(self.triangulated_graph.neighbors(n))) == 0:
                overflow += len(domains[n]) - max_eval / self.triangulated_graph.number_of_nodes()
        return overflow

    def init_graph_helpers(self):

        # Here we will decide if we want to recalculate the helper graphs
        edges_set = set([frozenset(x) for x in self.graph_function.graph.edges()])
        if self.edges_set != None and self.edges_set == edges_set:
            return

        self.graph = self.graph_function.graph
        self.edges_set = edges_set
        self.triangulated_graph = make_chordal(self.graph)
        self.clique_graph = build_clique_graph(self.graph)
        self.clique_tree = nx.minimum_spanning_tree(self.clique_graph)

class BruteForceAcquisitionOptimizer(AcquisitionOptimizer):
    """
    Class for optimizing the acquisition function in the simplest brute force way
    """
    def __init__(self, domain, X, mlflow_logging, max_eval, acq_opt_restarts=1):
        super(BruteForceAcquisitionOptimizer, self).__init__(domain)
        self.domain = domain
        self._domains = self.domain.combined_domain
        self.max_eval = max_eval
        self.acq_opt_restarts = acq_opt_restarts
        self.dimension = len(domain.combined_domain)
        
        # ==========================================
        self.evaluated_points = []
        for x in X: # we should have already picked at least one point randomly
                self.evaluated_points.append(x) 
        # ==========================================
        
        self.mlflow_logging = mlflow_logging
        self._total_cost = 0

    def optimize(self, f, f_df=None):
        x_best, fmin = None, float('inf')
        for _ in range(self.acq_opt_restarts):
            self._domains = self.domain.combined_domain
            if self.max_eval < -1:
                #logging.info("Zooming used")
                target_grid = -(self.max_eval)
                subdiv_count = 0
                while True:
                    is_stop = True
                    # Now we do the partitioning
                    domain_chunks = []
                    for domain in self._domains:
                        if len(domain) > target_grid:
                            # The case where you can chop it up
                            is_stop = False
                            chunk_len = math.ceil(len(domain) / float(target_grid))
                            domain_chunks.append(list(chunks(domain, chunk_len)))
                        else:
                            domain_chunks.append([ np.array([d]) for d in domain ])
                   
                    self._domains = [ [ np.random.choice(d, 1)[0] for d in domain_chunk] for domain_chunk in domain_chunks ]
                    x_best, fmin = self._optimize(f)
                    self._domains = []
                    
                    if is_stop:
                        break

                    # Reconstruct the 
                    for domain, x in zip(domain_chunks, x_best):
                        for d in domain:
                            if x in d:
                                self._domains.append(d)

                    subdiv_count += 1

                #self._domains = self.make_small_domains(self._domains, self.max_eval)
            else:
                #logging.info("Through optimization")
                assert(self.max_eval != 0)
                x_best, fmin = self._optimize(f)
        
        return x_best.reshape(1,-1), fmin

    def _optimize(self, f):
            
        subspace = np.array(list(itertools.product(*self._domains)))
        f_evals = f(subspace)
        best_ix = np.argmin(f_evals[:, 0])
        x_best = subspace[best_ix, :]
        f_min = f_evals[best_ix, 0]
        
        return x_best, f_min

class RestartAcquisitionOptimizer(AcquisitionOptimizer):
    def __init__(self, base_optimizer, acq_opt_restarts=1):
        self.base_optimizer = base_optimizer
        self.acq_opt_restarts = acq_opt_restarts

    def optimize(self, f, f_df):
        xbest, fbest = None, float('inf')
        for _ in range(self.acq_opt_restarts):
            x, fval = self.base_optimizer.optimize(f, f_df)

            if fval < fbest:
                xbest = x
                fbest = fval
        
        return xbest, fbest
