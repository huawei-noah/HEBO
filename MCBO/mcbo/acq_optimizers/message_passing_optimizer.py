import copy
import itertools
import random
import warnings
from functools import partial
from typing import Optional, Tuple, Any, Dict, List

import math
import networkx as nx
import numpy as np
import torch

from mcbo.acq_funcs.additive_lcb import AddLCB
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.models.gp.rand_decomposition_gp import RandDecompositionGP
from mcbo.search_space.params.bool_param import BoolPara
from mcbo.search_space.params.int_exponent_param import IntExponentPara
from mcbo.search_space.params.integer_param import IntegerPara
from mcbo.search_space.params.nominal_param import NominalPara
from mcbo.search_space.params.numeric_param import NumericPara
from mcbo.search_space.params.pow_integer_param import PowIntegerPara
from mcbo.search_space.params.pow_param import PowPara
from mcbo.search_space.params.sigmoid_param import SigmoidPara
from mcbo.search_space.search_space import SearchSpace
from mcbo.trust_region.tr_manager_base import TrManagerBase
from mcbo.trust_region.tr_utils import get_numdim_weights
from mcbo.utils.distance_metrics import hamming_distance
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


def trust_region_wrapper(x: torch.Tensor, f: AddLCB, tr_manager: TrManagerBase,
                         search_space: SearchSpace) -> torch.Tensor:
    mask = hamming_distance(
        x1=tr_manager.center[search_space.nominal_dims].to(x),
        x2=x[:, search_space.nominal_dims],
        normalize=False
    ) <= tr_manager.get_nominal_radius()
    assert torch.any(mask)
    out = f(x)

    out[~mask] = float('inf')

    return out


class MessagePassingOptimizer(AcqOptimizerBase):
    color_1: str = get_color(ind=5, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return MessagePassingOptimizer.color_1

    @property
    def name(self) -> str:
        return 'MP'

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: RandDecompositionGP,
                 acq_func: AddLCB,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:

        assert self.input_constraints is None or len(
            self.input_constraints) == 0, "Message Passing does not support input constraints"
        assert n_suggestions == 1, "Message Passing does not support multiple recommendations"

        graph = nx.empty_graph(n=self.search_space.num_dims)
        graph.add_edges_from([clique for clique in model.graph if len(clique) > 1])
        self.kwargs.update(kwargs)

        weights = None
        if tr_manager is not None:
            is_numeric = self.search_space.num_numeric > 0
            is_mixed = is_numeric and self.search_space.num_nominal > 0
            if is_numeric:
                weights = get_numdim_weights(
                    num_dim=self.search_space.num_numeric, is_numeric=is_numeric, is_mixed=is_mixed, kernel=model.kernel
                )
            else:
                weights = None

        optimizer = _MPOptimizer(
            domain=self.search_space,
            graph=graph,
            X=x_observed,
            max_eval=self.kwargs.get("max_eval", -4),
            acq_opt_restarts=self.kwargs.get("acq_opt_restarts", 1),
            tr_manager=tr_manager,
            weights=weights,
            model=model
        )

        f = {
            tuple(clique): partial(acq_func.partial_evaluate, clique=clique, model=model, t=x_observed.shape[0],
                                   **acq_evaluate_kwargs)
            for clique in model.graph
        }

        if tr_manager is not None and len(self.search_space.nominal_dims) > 0:
            f = {
                tuple(clique): partial(trust_region_wrapper, f=clique_f, tr_manager=tr_manager,
                                       search_space=self.search_space)
                for clique, clique_f in f.items()
            }

        x_best, fmin, total_mp_cost = optimizer.optimize(
            f, partial(acq_func.evaluate, model=model, **acq_evaluate_kwargs)
        )

        return x_best


def chunks(domain: list, n: int) -> list:
    """Taken from https://github.com/eric-vader/HD-BO-Additive-Models/blob/master/hdbo/acquisition_optimizer.py
    
    MIT License

    Copyright (c) 2020 Eric Han

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

    Yield successive n-sized chunks from domain."""
    for i in range(0, len(domain), n):
        yield domain[i:i + n]


def make_chordal(bn: nx.Graph):
    """
    Taken from https://github.com/eric-vader/HD-BO-Additive-Models/blob/master/hdbo/acquisition_optimizer.py
    
    MIT License

    Copyright (c) 2020 Eric Han

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

    This function creates a chordal graph - i.e. one in which there
    are no cycles with more than three nodes.

    Algorithm from Cano & Moral 1990 ->
    'Heuristic Algorithms for the Triangulation of Graphs'
    """
    chordal_E = list(bn.edges())

    # if moral graph is already chordal, no need to alter it
    if not nx.is_chordal(bn):
        temp_E = copy.copy(chordal_E)
        temp_V = []

        temp_G = nx.Graph()
        temp_G.add_edges_from(chordal_E)
        degree_dict = temp_G.degree()
        temp_V = [v for v, d in sorted(degree_dict, key=lambda x: x[1])]
        for v in temp_V:
            # Add links between the pairs nodes adjacent to Node i
            # Add those links to chordal_E and temp_E
            adj_v = set([n for e in temp_E for n in e if v in e and n != v])
            for a1 in adj_v:
                for a2 in adj_v:
                    if a1 != a2:
                        if [a1, a2] not in chordal_E and [a2, a1] not in chordal_E:
                            chordal_E.append([a1, a2])
                            temp_E.append([a1, a2])
            # remove Node i from temp_V and all its links from temp_E 
            temp_E2 = []
            for edge in temp_E:
                if v not in edge:
                    temp_E2.append(edge)
            temp_E = temp_E2

    G = nx.Graph()
    G.add_nodes_from(bn.nodes())
    G.add_edges_from(chordal_E)
    return G


def build_clique_graph(G: nx.Graph):
    '''
    Taken from https://github.com/eric-vader/HD-BO-Additive-Models/blob/master/hdbo/acquisition_optimizer.py
    
    MIT License

    Copyright (c) 2020 Eric Han

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
    '''
    clique_graph = nx.Graph()
    max_cliques = list(nx.chordal_graph_cliques(make_chordal(G)))
    # The case where there is only 1 max_clique
    if len(max_cliques) == 1:
        clique_graph.add_node(max_cliques.pop())
        return clique_graph

    for c1, c2 in itertools.combinations(max_cliques, 2):
        intersect = c1.intersection(c2)
        if len(intersect) != 0:
            # we put a minus sign because networkx only allows for MINIMUM Spanning Trees...
            clique_graph.add_edge(c1, c2, weight=-len(intersect))
        else:
            clique_graph.add_node(c1)
            clique_graph.add_node(c2)
    return clique_graph


# Fast join
sorted_tuple = lambda x: tuple(sorted(x))


class _MPOptimizer():
    """
    Class for optimizing the acquisition function in the considering that is of the form : f(x1,x2,x3,x4) = f1(x1,x2) + f2(x3) + f3(x4)
    Taken from https://github.com/eric-vader/HD-BO-Additive-Models/blob/master/hdbo/acquisition_optimizer.py
    
    MIT License

    Copyright (c) 2020 Eric Han

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

    def __init__(self, domain: SearchSpace, graph: nx.Graph, X: torch.Tensor, max_eval: int, acq_opt_restarts: int = 1,
                 grid_size: int = 150, tr_manager: Optional[TrManagerBase] = None,
                 weights: Optional[torch.Tensor] = None, model: Optional[RandDecompositionGP] = None):
        self.domain = domain
        self.max_eval = max_eval
        self.graph = graph
        self.acq_opt_restarts = acq_opt_restarts
        self.dimension = domain.num_params
        self.grid_size = grid_size

        self.edges_set = None
        self.mlflow_logging = None
        self.init_graph_helpers()

        # ==========================================
        self.evaluated_points = []
        for x in X:  # we should have already picked at least one point randomly
            self.evaluated_points.append(x)
            # ==========================================

        self._total_cost = 0
        self.tr_manager = tr_manager

        self.weights = weights
        self.model = model

    def _optimize(self, f: AddLCB) -> Tuple[np.ndarray, float]:

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

    def _optimize_helper(self, f: AddLCB) -> Tuple[np.ndarray, float]:
        # Optimizes each acquisition function on all disconnected trees
        x_best = np.array([-1] * self.dimension, dtype=np.float)
        fmin = 0
        # disconnectedComp = list(nx.components.connected.connected_component_subgraphs(self.clique_tree))
        disconnectedComp = [self.clique_tree.subgraph(c) for c in nx.connected_components(self.clique_tree)]
        np.random.shuffle(disconnectedComp)

        for clique_tree in disconnectedComp:  # loop on all disconnected trees
            fmin_tree, x_best_tree = self.optimize_connected_tree(clique_tree)
            x_best[x_best_tree != -1] = x_best_tree[x_best_tree != -1]

            fmin += fmin_tree

        return x_best, fmin

    def optimize(self, f: dict, full_f: AddLCB) -> Tuple[np.ndarray, float, float]:

        self._total_cost = 0
        self._f = f

        self.init_graph_helpers()

        # Original code
        # =============
        # Make the domain
        params = [self.domain.params[pname] for pname in self.domain.param_names]
        self.nominal_dims = []
        self.default_nominal_vals = []

        self._original_domains = []
        self._domains = []
        numerical_dim_no = 0
        if self.tr_manager is not None:
            x_centre = self.tr_manager.center

        for no, param in enumerate(params):
            if type(param) in {SigmoidPara, PowPara, NumericPara, IntExponentPara, IntegerPara, PowIntegerPara}:
                LB, UB = 0, 1

            if self.tr_manager is not None and type(param) not in {BoolPara, NominalPara}:
                assert self.model.kernel.numeric_dims[numerical_dim_no] == no
                new_UB = torch.clip(
                    x_centre[no] + self.weights[numerical_dim_no].to(x_centre) * self.tr_manager.radii['numeric'] / 2.0,
                    LB, UB).item()
                new_LB = torch.clip(
                    x_centre[no] - self.weights[numerical_dim_no].to(x_centre) * self.tr_manager.radii['numeric'] / 2.0,
                    LB, UB).item()
                LB = new_LB
                UB = new_UB
                numerical_dim_no += 1

            if type(param) in {SigmoidPara, PowPara, NumericPara}:
                # parameter is a real value
                incr = (UB - LB) / (self.grid_size - 1)
                domain = [LB + i * incr for i in range(self.grid_size)]
            elif type(param) in {IntExponentPara, IntegerPara, PowIntegerPara}:
                # parameter is an integer
                incr = (UB - LB) / (int(param.ub - param.lb))
                domain = [LB + i * incr for i in range(int(param.ub - param.lb) + 1)]
            elif type(param) == BoolPara:
                # parameter is a boolean
                self.nominal_dims.append(no)
                self.default_nominal_vals.append(x_centre[no].item() if self.tr_manager is not None else -1)
                domain = [0, 1]
            elif type(param) == NominalPara:
                # parameter is categorical
                self.nominal_dims.append(no)
                self.default_nominal_vals.append(x_centre[no].item() if self.tr_manager is not None else -1)
                domain = [i for i in range(len(param.categories))]
            else:
                raise NotImplementedError

            self._domains.append(domain)
            self._original_domains.append(domain)

        if self.max_eval > 0:
            # logging.info("T_Max used")
            self._domains = self.make_small_domains(self._domains, self.max_eval)
            x_best, fmin = self._optimize(f)
        else:
            if self.max_eval < -1:
                # logging.info("Zooming used")
                target_grid = -(self.max_eval)
                subdiv_count = 0
                while True:
                    is_stop = True
                    # Now we do the partitioning
                    domain_chunks = []
                    for i, domain in enumerate(self._domains):
                        if len(domain) > target_grid and not type(params[i]) in {NominalPara, BoolPara}:
                            # The case where you can chop it up
                            # do not use zooming for ordinal parameters
                            is_stop = False
                            chunk_len = math.ceil(len(domain) / float(target_grid))
                            domain_chunks.append(list(chunks(domain, chunk_len)))
                        else:
                            domain_chunks.append([[d] for d in domain])
                    self._domains = [[np.random.choice(d, 1)[0] for d in domain_chunk] for domain_chunk in
                                     domain_chunks]
                    x_best, fmin = self._optimize(f)
                    self._domains = []

                    # Reconstruct the 
                    temp_no = 0
                    for domain, x in zip(domain_chunks, x_best):
                        if type(params[temp_no]) in {NominalPara, BoolPara}:
                            self._domains.append([d[0] for d in domain])
                        else:
                            for d in domain:
                                if x in d:
                                    self._domains.append(d)
                                    break

                        temp_no += 1

                    subdiv_count += 1

                    if is_stop:
                        break

                # self._domains = self.make_small_domains(self._domains, self.max_eval)
            else:
                # logging.info("Through optimization")
                assert (self.max_eval != 0)
                x_best, fmin = self._optimize(f)

        # Domain sensitive
        # ==========================================
        pert_tries = 0
        while x_best.tolist() in [item.tolist() for item in self.evaluated_points] and pert_tries < 10:
            # Selected point already evaluated, performing perturbation
            pert_tries += 1
            var = random.randint(0, len(x_best) - 1)
            up = random.randint(0, 1)
            i = np.where(self._domains[var] == x_best[var])[0][0]
            if up == 1 and i < len(self._domains[var]) - 1:
                x_best[var] = self._domains[var][i + 1]
            elif i > 0:
                x_best[var] = self._domains[var][i - 1]

        if pert_tries == 10:
            print("Choosing random point")
            for var in range(len(x_best)):
                x_best[var] = random.choice(self._original_domains[var])

        x_best = np.array([x_best])
        if self.mlflow_logging != None:
            self.mlflow_logging.log_cost(self._total_cost)

        if pert_tries < 10:
            # ==========================================

            # ============================================================
            # Basic checks
            x_best = torch.tensor(x_best.tolist())
            full_fmin = full_f(x_best).item()
            if abs(full_fmin - fmin) > 0.01 * abs(full_fmin) and (
                    len(self.nominal_dims) == 0 or self.tr_manager is None):
                warnings.warn(
                    f"Cannot verify the correctness of message passing, {full_fmin}!={fmin}. Unless this is a result of previous exception, this likely indicates the acqusition function is not additive and should not be optimised with message passing.")
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
    def get_subspace(self, subspace_var_order: list) -> itertools.product:
        # subspace_var_order must be list
        subdomains = [self._domains[i] for i in subspace_var_order]
        return itertools.product(*subdomains)

    def optimize_connected_tree(self, tree: nx.Graph) -> Tuple[np.ndarray, float]:
        self._computed_cliques_G = set()

        # We unroll the first call for message_passing, by starting with the max degree node
        root = max(tree.degree(), key=lambda x: x[1])[0]
        children = list(self.clique_tree.neighbors(root))

        input_messages = self.broadcast_mp_children(root, children)

        intersection = root - frozenset().union(*children)
        marginal = root - intersection

        # Unpack
        marginal_var_order = sorted(marginal)

        if len(intersection) == 0:
            # Special case where we need to optimize directly for the outstanding variables
            filtered_cliques_G = self.filter_cliques_G(root)

            new_X = np.full((1, self.domain.num_params), -1, dtype=np.float)
            new_X[:, self.nominal_dims] = self.default_nominal_vals

            fmin_overall, x_best_overall = self.optimize_discrete(new_X, children, marginal_var_order, input_messages,
                                                                  root, filtered_cliques_G)
            fmin_overall = fmin_overall[0]
            x_best_overall = x_best_overall[0]
        else:

            # We ignore the message, as we are at the root
            fmin_overall, x_best_overall, _ = self.compute_message(intersection, children, marginal_var_order,
                                                                   input_messages, root)

        # Returns the element for which the value function is minimal (the min function selects with respect to the first value of the list)        
        return fmin_overall, x_best_overall

    def broadcast_mp_children(self, root: int, children: list) -> list:
        input_messages = []
        for child in children:
            child_messages = self.message_passing(child, root)
            input_messages.append(child_messages)
        return input_messages

    def compute_message(self, intersection: list, children: list, marginal_var_order: list, input_messages: np.ndarray,
                        root: int) -> Tuple[np.ndarray, float, np.ndarray]:

        # Filter cliques for computational efficiency
        filtered_cliques_G = self.filter_cliques_G(root)

        # Unpack
        intersection_var_order = sorted(intersection)
        # logging.info(f'intersection_var_order {intersection_var_order}')
        subspace_x = np.array(list(self.get_subspace(intersection_var_order)))
        new_X = np.full((subspace_x.shape[0], self.domain.num_params), -1, dtype=np.float)
        new_X[:, self.nominal_dims] = self.default_nominal_vals
        new_X[:, intersection_var_order] = subspace_x

        fmin_curr, x_best_curr = self.optimize_discrete(new_X, children, marginal_var_order, input_messages, root,
                                                        filtered_cliques_G)

        # Compose the message
        output_message = {tuple(sub_x): (fmin_ea, x_best_ea) for sub_x, fmin_ea, x_best_ea in
                          zip(subspace_x, fmin_curr, x_best_curr)}

        _fmin_i = np.argmin(fmin_curr)
        fmin_overall = fmin_curr[_fmin_i]
        x_best_overall = np.copy(x_best_curr[_fmin_i])

        self._computed_cliques_G.update(filtered_cliques_G)
        return fmin_overall, x_best_overall, output_message

    def filter_cliques_G(self, root: int) -> set:
        subgraph_G = self.graph.subgraph(root)
        cliques_G = nx.find_cliques(subgraph_G)
        # Slightly more efficient?
        # [ s for s in map(frozenset, cliques_G) if not s in self._computed_cliques_G ]
        # return set(map(tuple, map(sorted, cliques_G))) - self._computed_cliques_G
        return set(map(sorted_tuple, cliques_G)) - self._computed_cliques_G

    def message_passing(self, root: int, parent: int) -> np.ndarray:
        # For all values of the variables in intersection(root, parent), returns
        # the minimum value of the acquisition function for the variables in root-parent,
        # and the associated x for the relevant variables (0 for the others).

        children = list(self.clique_tree.neighbors(root))
        children.remove(parent)  # note that neighbours would include parents

        input_messages = self.broadcast_mp_children(root, children)

        intersection = root.intersection(parent)  # Notice that len(intersection) != 0
        marginal = root - intersection

        # Unpack        
        marginal_var_order = sorted(marginal)

        # Quantities to return when optimizing the root
        fmin_overall, x_best_overall, output_message = self.compute_message(intersection, children, marginal_var_order,
                                                                            input_messages, root)

        return output_message

    def optimize_discrete(self, new_X: np.ndarray, children: list, marginal_var_order: list, input_messages: np.ndarray,
                          root: int, filtered_cliques_G: list) -> Tuple[float, np.ndarray]:
        fmin_X = np.full(new_X.shape[0], np.inf, dtype=np.float)
        # fmin_X = []
        X_best = np.full((new_X.shape[0], self.domain.num_params), np.inf, dtype=np.float)
        # x_best = None

        # Prepare the intersection_child_var_order
        # Unpack
        intersection_child_var_orders = [sorted(root.intersection(c)) for c in children]

        # Unpack
        for x_marginal in self.get_subspace(marginal_var_order):
            new_X[:, marginal_var_order] = x_marginal

            f_X_marginal = self.load_messages(new_X, input_messages,
                                              intersection_child_var_orders)  # also modifies new_x

            # 2- Compute overall function for this node and optimise it
            f_X_marginal += self.compute_node_function(filtered_cliques_G, new_X)

            self._total_cost += new_X.shape[0]

            fmin_update_indices = fmin_X > f_X_marginal
            X_best[fmin_update_indices] = new_X[fmin_update_indices]
            fmin_X[fmin_update_indices] = f_X_marginal[fmin_update_indices]

        return fmin_X, X_best

    def load_messages(self, new_X: np.ndarray, input_messages: np.ndarray,
                      intersection_child_var_orders: list) -> np.ndarray:
        # modify new_x with messages received from 'children' and returns the sum of all function values of the children nodes
        f_x_marginal = np.zeros(new_X.shape[0])
        for m, intersection_child_var_order in zip(input_messages, intersection_child_var_orders):
            marginal_x_child = new_X[:, intersection_child_var_order]
            fmin_child, best_x_child = map(np.array,
                                           zip(*map(lambda sub_x_child: m[tuple(sub_x_child)], marginal_x_child)))

            # Update the marginals and x
            # replace 0 values in new_x with ones in best_x_child
            f_x_marginal += fmin_child
            new_X[best_x_child != -1] = best_x_child[best_x_child != -1]

        return f_x_marginal

    def compute_node_function(self, filtered_cliques_G: list, new_X: np.ndarray) -> float:
        fval = np.zeros(new_X.shape[0])
        # For all cliques c of the subset of the triangulated clique 
        # if clique_G in self._f:
        for clique_G in filtered_cliques_G.intersection(self._f.keys()):
            new_fval = self._f[clique_G](torch.tensor(new_X.tolist())).detach().cpu().numpy().reshape(-1)
            fval += new_fval
            # Cases that will be excluded here, 
            # 1. cliques that does not have any computable edges
            # 2. cliques that are partially cut off
            # 3. cliques that are previous computed
            # print(clique)
            # assert(False)
        return fval

    def make_small_domains(self, domains: list, max_eval: int) -> Dict[Any, Any]:
        small_domains = {}
        # disconnectedComp = list(nx.components.connected.connected_component_subgraphs(self.triangulated_graph))
        disconnectedComp = [self.triangulated_graph.subgraph(c) for c in
                            nx.connected_components(self.triangulated_graph)]
        nDiscComp = len(disconnectedComp)
        max_eval += self.overflow(domains, max_eval)
        for subgraph in disconnectedComp:
            N_subgraph = max_eval * subgraph.number_of_nodes() / self.triangulated_graph.number_of_nodes()  # number of evaluations for this component
            n_clique_var = 0
            for clique in nx.find_cliques(subgraph):
                n_clique_var += len(clique)
            for var in subgraph.nodes():
                clique_size = nx.node_clique_number(subgraph, var)
                N_clique = clique_size * N_subgraph / n_clique_var  # number of evaluation for this clique
                N_var = max(int(round(N_clique ** (1.0 / clique_size))), 2)
                small_domains[var] = self.choose_from(domains[var], N_var)
        return small_domains

    def choose_from(self, domain: List, N_var: int) -> list:
        # randomly selects N_var DIFFERENT points
        if N_var >= len(domain):
            return np.array(domain)
        # Cannot be replaced, items drawn must be unique
        return np.random.choice(list(domain), N_var, replace=False)

    def overflow(self, domains: list, max_eval: int) -> float:
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
        edges_set = set([frozenset(x) for x in self.graph.edges()])
        if self.edges_set != None and self.edges_set == edges_set:
            return

        self.graph = self.graph
        self.edges_set = edges_set
        self.triangulated_graph = make_chordal(self.graph)
        self.clique_graph = build_clique_graph(self.graph)
        self.clique_tree = nx.minimum_spanning_tree(self.clique_graph)
