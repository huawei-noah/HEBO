import numpy as np
from scipy import signal
from pymoo.core.problem import Problem
from functools import partial
import warnings

corruption_params = {
    'large': {"a0": -0.03, "a1": 0.20, "a2": 0.16, "a3": 0.06},
    'small': {"a0": -0.03, "a1": 0.05, "a2": 0.08, "a3": 0.03},
}


def small_corruption_func(x):
    return _corruption(x, **corruption_params['small'])


def large_corruption_func(x):
    return _corruption(x, **corruption_params['large'])


def _corruption(x, a0, a1, a2, a3):
    assert np.all(x >= 0)
    assert np.all(x <= 1)
    a = 0.0 + 1.0 * signal.square(4 * 2 * np.pi * x)
    b = 0.5 + 0.5 * signal.square(4 * 2 * np.pi * x)
    base = a * b
    p0 = 0.3 * np.pi
    p1 = 0
    p2 = np.pi
    p3 = 0.5 * np.pi
    s0 = a0 * signal.sawtooth(p0 + 15 * 2 * np.pi * x)
    s1 = a1 * signal.sawtooth(p1 + 10 * 2 * np.pi * x)
    s2 = a2 * signal.sawtooth(p2 + 30 * 2 * np.pi * x)
    s3 = a3 * signal.sawtooth(p3 + 40 * 2 * np.pi * x)
    return base * (s0 + s1 + s2 + s3)


def normalise(v, lower_limits, ranges):
    return (v - lower_limits) / ranges


def corrupted_func(v, func, f_range, corruption_func, lower_limits, ranges):
    return func(v) + f_range * np.max(
        [corruption_func(v_dim_norm) for v_dim_norm in normalise(v, lower_limits, ranges)],
        axis=1, keepdims=True
    )


def add_corruption(func, bounds, f_min, f_max, corruption_func):
    f_range = f_max - f_min
    lower_limits = np.array([b[0] for b in bounds])
    upper_limits = np.array([b[1] for b in bounds])
    ranges = upper_limits - lower_limits
    crpt_f = partial(corrupted_func, func=func, f_range=f_range, corruption_func=corruption_func,
                     lower_limits=lower_limits, ranges=ranges)
    return crpt_f


class TestFunctions(Problem):
    def __init__(self, n_var: int, n_obj: int, n_constr: int, xl: np.array, xu: np.array,
                 raw_func = lambda x: x, raw_parallel_eval: bool = True,
                 raw_f_min: float = None, raw_f_max: float = None, minimization: bool = True,
                 input_range_offset: float = None, x_offset: float = 7,
                 crpt_method: str = 'raw', mesh_granularity: int = 300, name: str = "crypt_prob",
                 **kwargs):
        super(TestFunctions, self).__init__(n_var, n_obj, n_constr, xl, xu)
        self.name = f"{name}-{n_var}D-{crpt_method}"
        self.n_var = n_var
        self.raw_func = raw_func
        self.crpt_method = crpt_method
        self.mesh_coordinate_num = mesh_granularity
        self.input_range_offset = input_range_offset
        self.x_offset = x_offset
        self.min_value = raw_f_min
        self.max_value = raw_f_max
        self.minimization = minimization
        self.raw_parallel_eval = raw_parallel_eval

        self.input_bounds = [(xl[i], xu[i]) for i in range(len(xl))]
        self.input_bounds_offset = [
            (xl[i] - self.input_range_offset, xu[i] - self.input_range_offset)
            for i in range(len(xl))
        ]

        # mesh coordinates and values
        self.x_coords = None
        self.mesh_coords = None
        self.raw_mesh_vals = None
        if n_var == 1:
            self.x_coords = [np.linspace(self.input_bounds_offset[0][0],
                                         self.input_bounds_offset[0][1],
                                         self.mesh_coordinate_num),
                             ]
            self.mesh_coords = self.x_coords[0].reshape(-1, 1)
        elif n_var <= 2:
            self.x_coords = np.meshgrid(
                *[
                    np.linspace(self.input_bounds_offset[d_i][0],
                                self.input_bounds_offset[d_i][1],
                                self.mesh_coordinate_num)
                    for d_i in range(n_var)
                ]
            )
            self.mesh_coords = np.vstack([c.flat for c in self.x_coords]).T
        else:
            # no need mesh for n_var>2 since we do not plot and the mesh number grows exponentially
            # with the dimension
            self.x_coords = None
            self.mesh_coords = None

        if self.mesh_coords is not None:
            if self.raw_parallel_eval is True:
                self.raw_mesh_vals = self.raw_func(
                    self.mesh_coords + self.x_offset + self.input_range_offset
                )
            else:
                self.raw_mesh_vals = np.array(
                    [self.raw_func(c) for c in self.mesh_coords]
                ).reshape(-1, 1)
        else:
            self.mesh_vals = None

        # automatic estimation of f_min/f_max
        if (raw_f_min is None or raw_f_max is None) and self.n_var > 2:
            # raise ValueError("Only support automatic max/min estimation for n_var<=2")
            warnings.warn("Only support automatic max/min estimation for n_var<=2")
        else:
            if raw_f_max is None:
                raw_f_max = self.raw_mesh_vals.max() + 0.01
            if raw_f_min is None:
                raw_f_min = self.raw_mesh_vals.min() - 0.01

        # add corruption
        if crpt_method == "raw":
            self.corrupted_func = self.raw_func
            self.mesh_vals = self.raw_mesh_vals
            self.min_value = raw_f_min
            self.max_value = raw_f_max
        elif crpt_method in ['small', 'large']:
            crpt = small_corruption_func if crpt_method == 'small' else large_corruption_func
            self.corrupted_func = add_corruption(
                self.raw_func, self.input_bounds_offset, raw_f_min, raw_f_max, crpt
            )
            self.mesh_vals = self.corrupted_func(self.mesh_coords + self.input_offset) \
                if self.mesh_coords is not None else None
            self.min_value = raw_f_min - abs(raw_f_max - raw_f_min) * sum(
                corruption_params[crpt_method].values())
            self.max_value = raw_f_max + abs(raw_f_max - raw_f_min) * sum(
                corruption_params[crpt_method].values())
        else:
            raise ValueError("Invalid corruption:", crpt_method)

        self.kwargs = kwargs

    def _evaluate(self, X, *args, return_values_of="auto", return_as_dictionary=False, **kwargs):
        neg_eval = kwargs.get('neg_eval', False)
        _a = -1.0 if neg_eval else 1.0
        inputs = X + self.input_range_offset + self.x_offset
        if self.raw_parallel_eval is True:
            ret = self.corrupted_func(inputs)
        else:
            ret = np.array(
                [self.corrupted_func(c) for c in inputs]
            ).reshape(-1, 1)
        return ret * _a

    def evaluate(self, X, *args, return_values_of="auto", return_as_dictionary=False, **kwargs):
        return self._evaluate(X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utils.visulaization as vis

    n_var = 1
    input_bounds = [(-10, +10), ] * n_var
    offset = 3
    func_name = "CustomK"
    raw_func = get_test_problem(func_name, n_var=n_var).evaluate
    # raw_func = lambda x: np.sin(x).mean()
    prob = TestFunctions(n_var, n_obj=1, n_constr=0,
                         xl=[input_bounds[i][0] for i in range(n_var)],
                         xu=[input_bounds[i][1] for i in range(n_var)],
                         raw_func=raw_func, raw_f_min=-1, raw_f_max=1,
                         crpt_method='large',
                         x_offset=offset, input_range_offset=100,
                         mesh_granularity=5000,
                         name=func_name)
    if prob.n_var <= 2:
        vis.visualize_problem(prob)
        plt.show()
