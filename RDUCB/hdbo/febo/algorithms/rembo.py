from itertools import cycle
import logging
from febo.algorithms import Algorithm, AlgorithmConfig, ModelMixin
from febo.environment import ContinuousDomain
from febo.solvers import ScipySolver
from febo.utils.config import ConfigField, config_manager, assign_config
from febo.utils import join_dtype_arrays, join_dtypes

import numpy as np

class RemboConfig(AlgorithmConfig):
    emb_d = ConfigField(2, comment='subspace dimension')
    _section = 'algorithm.rembo'

config_manager.register(RemboConfig)


@assign_config(RemboConfig)
class Rembo(ModelMixin, Algorithm):
    """
    Code is based on https://github.com/jmetzen/bayesian_optimization

    """

    def initialize(self, **kwargs):
        # compute embedded domain
        domain = kwargs.get('domain')
        self.n_dims = domain.d
        self.n_embedding_dims = self.config.emb_d
        logging.info(f"Subspace dim {self.n_embedding_dims}")

        # Determine random embedding matrix
        self.A = np.random.normal(size=(self.n_dims, self.n_embedding_dims))
        self._boundaries = np.array([[l,u] for l,u in zip(domain.l, domain.u)])
        # Compute boundaries on embedded space
        self._boundaries_embedded = self._compute_boundaries_embedding(self._boundaries)
        self._embbeded_domain = ContinuousDomain(l=self._boundaries_embedded[:,0], u=self._boundaries_embedded[:,1])
        self._embbded_solver = ScipySolver(self._embbeded_domain)

        # set model_domain to embbeded_domain
        kwargs['model_domain'] = self._embbeded_domain

        # We manually add x_emb into the data for initial_data        
        A_inv = np.linalg.pinv(self.A)
        
        # Init the x_emb in inital_data
        initial_data = kwargs['initial_data']
        aug_initial_data = []
        add_data = np.empty(shape=(), dtype=self.dtype)
        for ea_d in initial_data:
            add_data['x_emb'] = A_inv.dot(ea_d['x'])
            
            aug_data_dtype = join_dtypes(ea_d.dtype, add_data.dtype)
            ea_aug_data = join_dtype_arrays(ea_d, add_data, aug_data_dtype).view(np.recarray)
            
            aug_initial_data.append(ea_aug_data)
        kwargs['initial_data'] = aug_initial_data
        super().initialize(**kwargs)
        



        # self.data_space = data_space
        # if self.data_space is not None:
        #     self.data_space = np.asarray(self.data_space)
        #     if self.data_space.shape[0] != self.n_dims - n_keep_dims:
        #         raise Exception("Data space must be specified for all input "
        #                         "dimensions which are not kept.")



    def _acquisition_function(self, X):
        """ works in embedded domain """
        return -self.model.ucb(X)

    def _next(self):
        X_query_embedded = self._embbded_solver.minimize(self._acquisition_function)[0]

        # Map to higher dimensional space
        # it is clip to hard boundaries by base class
        X_query = self._map_to_dataspace(X_query_embedded)
        return X_query, {'x_emb' : X_query_embedded}

    def _map_to_dataspace(self, X_embedded):
        """ Map data from manifold to original data space. """
        return self.A.dot(X_embedded)
        # if self.data_space is not None:
        #     X_query_kd = (X_query_kd + 1) / 2 \
        #                  * (self.data_space[:, 1] - self.data_space[:, 0]) \
        #                  + self.data_space[:, 0]
        # X_query = np.hstack((X_embedded[:self.n_keep_dims], X_query_kd))

    def _compute_boundaries_embedding(self, boundaries):
        """ Approximate box constraint boundaries on low-dimensional manifold"""
        # # Check if boundaries have been determined before
        # boundaries_hash = hash(boundaries[self.n_keep_dims:].tostring())
        # if boundaries_hash in self.boundaries_cache:
        #     boundaries_embedded = \
        #         np.array(self.boundaries_cache[boundaries_hash])
        #     boundaries_embedded[:self.n_keep_dims] = \
        #         boundaries[:self.n_keep_dims]  # Overwrite keep-dim's boundaries
        #     return boundaries_embedded

        # Determine boundaries on embedded space
        boundaries_embedded = \
            np.empty((self.n_embedding_dims, 2))
        for dim in range(self.n_embedding_dims):
            x_embedded = np.zeros(self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                        x < boundaries[:, 0],
                        x > boundaries[:, 1])) \
                        > self.n_dims / 2:
                    break
                x_embedded[dim] -= 0.01
            boundaries_embedded[dim, 0] = x_embedded[dim]

            x_embedded = np.zeros( self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                        x < boundaries[:, 0],
                        x > boundaries[:, 1])) \
                        > self.n_dims/ 2:
                    break
                x_embedded[dim] += 0.01
            boundaries_embedded[dim, 1] = x_embedded[dim]

        # self.boundaries_cache[boundaries_hash] = boundaries_embedded

        return boundaries_embedded

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        fields += [('x_emb', f'({self.config.emb_d},)f')]
        return fields

    def _get_x_from_data(self, data):
        return data['x_emb']

    def best_predicted(self):
        best_x_emb, self._best_predicted_y = self._embbded_solver.minimize(lambda X : -self.model.mean(X))
        self._best_predicted_x = self.domain.project(self._map_to_dataspace(best_x_emb))
        return self._best_predicted_x

class InterleavedRemboConfig(AlgorithmConfig):
    interleaved_runs = ConfigField(4)
    _section = 'algorithm.rembo'

@assign_config(InterleavedRemboConfig)
class InterleavedRembo(Algorithm):
    def __init__(self, **kwargs):
        self._rembos = [Rembo(**kwargs) for _ in range(self.config.interleaved_runs)]
        super().__init__(**kwargs)

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        for rembo in self._rembos:
            rembo.initialize(**kwargs)
            rembo._best_predicted_computed = False

        self._cycle_rembo = cycle(self._rembos)
        self._current_rembo = next(self._cycle_rembo)

    def next(self):
        return self._current_rembo.next()

    def add_data(self, data):
        self._current_rembo.add_data(data)
        self._current_rembo._best_predicted_computed = False
        self._current_rembo = next(self._cycle_rembo)

    def best_predicted(self):
        best_x = None
        best_y = -10e10
        for rembo in self._rembos:
            if not rembo._best_predicted_computed:
                rembo.best_predicted()
                rembo._best_predicted_computed = True

            if rembo._best_predicted_y > best_y:
                best_x = rembo._best_predicted_x
                best_y = rembo._best_predicted_y

        return best_x

    def _get_dtype_fields(self):
        return self._rembos[0]._get_dtype_fields()
