import numpy as np
from febo.environment import DiscreteDomain
from febo.environment.domain import ContinuousDomain


class TrustRegionDomain(DiscreteDomain):

    def __init__(self, domain, x0, radius, num_random_points=500, num_axis_points=10):
        self._domain = domain
        self.x0 = x0
        self.radius = radius
        axis_points, self._axis_point_list = self._get_axes_points(num_axis_points)
        random_points = self._get_random_points(num_random_points)
        points = np.vstack((axis_points, random_points))
        super().__init__(points, domain.d)

    def get_axis_points(self, coord):
        return self._axis_point_list[coord]

    def _get_axes_points(self, num_points):
        axis_point_list = []
        points = self.x0.copy().reshape(1,-1)

        for i, v in enumerate(np.eye(self._domain.d)):
            axis_points_1 = self.x0 + v * np.linspace(0, self.radius[i], num_points).reshape(-1, 1)[1:]
            axis_points_2 = self.x0 - v * np.linspace(0, self.radius[i], num_points).reshape(-1, 1)[1:]

            # remove points beyond boundary
            axis_points = np.vstack((axis_points_1, axis_points_2))
            axis_points = axis_points[np.logical_and((axis_points <= self._domain.u).all(axis=1), (axis_points >= self._domain.l).all(axis=1))]

            points = np.vstack((points, axis_points))
            axis_point_list.append(np.vstack((self.x0, axis_points)))

        return points, axis_point_list

    def _get_random_points(self, num_points):
        # generate random points in ball with radius
        directions = np.random.normal(0, 1, size=num_points * self._domain.d).reshape(num_points, self._domain.d)
        directions = directions / np.linalg.norm(directions, axis=1).reshape(-1, 1)
        return self._domain.project(self.x0 + directions * np.random.uniform(0, 1, size=num_points).reshape(-1, 1) * self.radius)

class LineDomain(ContinuousDomain):
    """
    This class represents a 1d subdomain.
    """

    def __init__(self, domain, x0, direction):
        self._domain = domain
        self.x0 = x0
        self.direction = direction / np.linalg.norm(direction)
        lower, upper = self._find_subdomain_bounds(domain)
        super(LineDomain, self).__init__(lower, upper)
        self.c = self.project_on_line(x0)[0]

    def project_on_line(self, X):
        X = np.atleast_2d(X)
        return np.dot(X - self.x0, self.direction.T)

    def embed_in_domain(self, X):
        X = np.atleast_2d(X)
        return np.dot(X, self.direction) + self.x0

    def _find_subdomain_bounds(self, domain):

        #make sure these are numpy arrays
        old_lower = np.array(domain._l)
        old_upper = np.array(domain._u)

        #define the output arrays
        lower = np.empty((1))
        upper = np.empty((1))

        for j in range(len(self.direction)):
            v = self.direction[j]
            if len(old_lower) != len(v) or len(old_upper) != len(v):
                raise ValueError("Basis needs to have the same dimension than the bounds")
            temp_l = np.empty_like(v)
            temp_u = np.empty_like(v)
            for i in range(len(v)):
                if v[i] > 0:
                    temp_u[i] = (old_upper[i]-self.x0[i])/v[i]
                    temp_l[i] = (old_lower[i]-self.x0[i])/v[i]
                elif v[i] < 0:
                    temp_l[i] = (old_upper[i]-self.x0[i])/v[i]
                    temp_u[i] = (old_lower[i]-self.x0[i])/v[i]
                else:
                    temp_l[i] = -np.inf
                    temp_u[i] = np.inf
            #we use the minimum distance to the boundaries to define our new bounds
            upper[j] = np.min(temp_u)
            lower[j] = np.max(temp_l)

        return lower, upper

    def set_u(self,new_u):
        self._u = new_u
        self._range = self._u - self._l

    def set_l(self,new_l):
        self._l = new_l
        self._range = self._u - self._l

class RandomSafeSubDomain(LineDomain):
    """
    This Class creates a D dimensional Domain inside a Safeset in a random direction
    """

    def __init__(self, domain, x0, direction, safeset):
        self._safeset = safeset
        super().__init__(domain, x0,direction)

    def _find_subdomain_bounds(self):
        if not(self._safeset.in_safeset(self.x0)):
            self.x0 = self._safeset.project_back_to_ellipse(self.x0)
        lower, upper = self._safeset.one_dim_bounds(self.direction[0], self.x0)
        return np.array([-lower]), np.array([upper])

