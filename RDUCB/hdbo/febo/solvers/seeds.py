import numpy as np
from febo.utils.config import ConfigField, Config, assign_config, Configurable, config_manager


class SeedsConfig(Config):
    expected_tail_length = ConfigField(100)
    max_tail_length = ConfigField(200)
    num_random_points = ConfigField(50)
    safe_projection = ConfigField(False)
    projection_max_line_search = ConfigField(10)
    _section = 'optimizers.seeds'

# config_manager.register(SeedsConfig)

@assign_config(SeedsConfig)
class Seeds(Configurable):
    """
    Class to track a set of seeds used as starting points for the continuous optimizers.
    TODO: Remove counter-intuitive behaviour of _next()
    """

    def __init__(self, domain, initial_safe_point=None):
        self.domain = domain
        self.d = domain.d
        self.initial_safe_point = initial_safe_point
        self.tail = np.empty((0, self.d))
        self.random_points = np.empty((0, self.d))
        self.initial_safe_point = initial_safe_point


    def next(self):
        """
        return next seed
        :return:
        """
        # randomly determine which list to choose

        #  pick randomly between the two lists, and increase current position of chosen list, cycle to beginning if needed
        if np.random.binomial(1, 0.5) and len(self.tail):
            x = self.tail[self.pos_tail]
            self.pos_tail = (self.pos_tail+ 1) % self.tail.shape[0]
        else:
            # TODO generated random points on demand / use up list, then generate new points
            x = self.random_points[self.pos_random_points]
            self.pos_random_points = (self.pos_random_points + 1) % self.random_points.shape[0]

        return x

    def _return_initial_point(self):
        self.next = self._next
        return self.initial_safe_point

    def new_iteration(self, sfun = None):
        self.pos_tail = 0
        self.pos_random_points = 0
        self.generate_random_points(sfun)

        if not self.initial_safe_point is None:
            self.add_to_tail(self.initial_safe_point)

    def add_to_tail(self, x):
        """
            append x to tail, and apply binomial mask to fade out older items
        """
        tail_length = self.tail.shape[0]
        if tail_length >= self.config.expected_tail_length:
            # flip a coin, such that in expectation, we have (SEEDS_TAIL_LENGTH_EXPECTED - 1) successes
            p = (self.config.expected_tail_length - 1)/tail_length
            mask = np.random.binomial(1, p, size=tail_length).astype(bool)
            self.tail = self.tail[mask] # boolean mask

        if self.tail.shape[0] > self.config.max_tail_length:
            self.tail = self.tail[0: self.config.max_tail_length]

        # insert new point on top
        self.tail = np.concatenate((x.reshape(-1, self.d), self.tail))

    def generate_random_points(self, sfun=None):
        """
        adds random points to the queue
        :param sfun: function, which takes array x as input, and returns True if x is safe and False otherwise.
        """
        self.random_points = self.domain.l + self.domain.range * np.random.uniform(size=(self.config.num_random_points, self.d))

        if self.config.safe_projection and sfun is not None:
            # sample a point from a known safe set
            point_array = np.concatenate((self.initial_safe_point.reshape(1, -1), self.tail))
            projected_points = []

            for point in self.random_points:
                random_index = int(np.random.randint(0, point_array.shape[0], 1))
                safe_point = point_array[random_index, :]
                # do a line search
                alpha = 1.0
                for i in range(self.config.projection_max_line_search):
                    new_point = safe_point + alpha * (point - safe_point)
                    if sfun(new_point):
                        break

                    alpha /= 1.5

                # just always add the last point we get, even if it is not safe
                projected_points.append(new_point)

            self.random_points = np.array(projected_points)