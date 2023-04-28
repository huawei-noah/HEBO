
"""
Class which reads the finite linear bandit from json file.
Allows to reproduce results from previous code base.
"""

# class FiniteLinearBandit(BenchmarkEnvironment):
#     """
#     Quick sketch of a finite linear bandit
#     """
#
#     def __init__(self, path=None):
#         super().__init__(path=path)
#         self._theta = np.ones(self.config.dimension)
#         self._theta = self._theta / np.linalg.norm(self._theta)
#
#         np.random.seed(self.seed)
#         self._domain_points = np.random.multivariate_normal(np.zeros(self.config.dimension),
#                                                             np.eye(self.config.dimension),
#                                                             size=self.config.num_domain_points)
#
#         np.random.seed()  # reset to random state
#
#         self._domain_points = self._domain_points / np.maximum(np.linalg.norm(self._domain_points, axis=-1),
#                                                                np.ones(self.config.num_domain_points)).reshape(-1, 1)
#         self._domain = DiscreteDomain(self._domain_points)
#         self._max_value = self._get_max_value()
#
#         import os
#         import json
#         json_file = os.path.join(self._path, 'environment.json')
#
#         if os.path.exists(json_file):
#             with open(json_file, 'r') as file:
#                 environment_config = json.load(file)
#                 for i, point in enumerate(environment_config['domain']):
#                     self._domain_points[i] = np.array(point[0])
#
#
#             self._domain = DiscreteDomain(self._domain_points)
#             self._max_value = self._get_max_value()
#             print("using environment.json")
#
#     @property
#     def name(self):
#         return "Finite Linear Bandit"
#
#     @property
#     def _requires_random_seed(self):
#         return True