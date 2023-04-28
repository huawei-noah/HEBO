
import hpolib.benchmarks.synthetic_functions as hpobench

from febo.environment import ContinuousDomain
from febo.environment.benchmarks import BenchmarkEnvironment
from hpolib.benchmarks import synthetic_functions
import numpy as np

class HpolibBenchmark(BenchmarkEnvironment):
    """
    Abstract class to convert Hpolib benchmarks.
    """
    def __init__(self, bench, path=None, min_value=-np.inf):
        super().__init__(path)
        self._bench = bench
        info = bench.get_meta_information()
        self._max_value = -info['f_opt']

        l = np.array([b[0] for b in info['bounds']])
        u = np.array([b[1] for b in info['bounds']])
        self._domain = ContinuousDomain(l, u)
        self._x0 = l + 0.1*self._domain.range
        self._min_value = min_value

    def f(self, x):
        return np.maximum(-self._bench(x), self._min_value)

class Branin(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Branin(), path, min_value=-2)

class Hartmann3(HpolibBenchmark):
    """
    d=3
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Hartmann3(), path)

class Hartmann6(HpolibBenchmark):
    """
    d=6
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Hartmann6(), path)
        self._x0 = np.array([0.1335990371483741, 0.2743781816448671, 0.2879962344461537, 0.10242147970254536, 0.3959197145814795, 0.5982863622683936])
        self._old_max_value = self._max_value
        self._max_value = 1.

    def f(self, X):
        return super().f(X)/self._old_max_value

class Camelback(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Camelback(), path)
        # overwrite domain to get a reasonable range of function values
        self._domain = ContinuousDomain(np.array([-2, -1]), np.array([2, 1]))

class Forrester(HpolibBenchmark):
    """
    d=1
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Forrester(), path)

class Bohachevsky(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Bohachevsky(), path)

class GoldsteinPrice(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.GoldsteinPrice(), path)

class Levy(HpolibBenchmark):
    """
    d=1
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Levy(), path)

class Rosenbrock(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.Rosenbrock(), path)

class Rosenbrock5D(HpolibBenchmark):
    """
    d=5
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.rosenbrock.Rosenbrock5D(), path)

class Rosenbrock10D(HpolibBenchmark):
    """
    d=10
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.rosenbrock.Rosenbrock10D(), path)

class Rosenbrock20D(HpolibBenchmark):
    """
    d=20
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.rosenbrock.Rosenbrock20D(), path)

class SinOne(HpolibBenchmark):
    """
    d=1
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.SinOne(), path)

class SinTwo(HpolibBenchmark):
    """
    d=2
    """
    def __init__(self, path=None):
        super().__init__(synthetic_functions.SinTwo(), path)