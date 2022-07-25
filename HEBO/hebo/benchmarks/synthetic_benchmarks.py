# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.distributions import MultivariateNormal
from pymoo.factory import get_problem
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import special_ortho_group

from hebo.design_space.design_space import DesignSpace

class BenchParamScaler:
    def __init__(self, lb : np.ndarray, ub : np.ndarray):
        self.scaler = MinMaxScaler((-1, 1)).fit(np.vstack([lb.reshape(-1), ub.reshape(-1)]))
    
    def transform(self, x : np.ndarray) -> np.ndarray:
        """
        Transform data from [lb, ub] to [-1, 1]
        """
        return self.scaler.transform(x)

    def inverse_transform(self, x : np.ndarray) -> np.ndarray:
        """
        Transform data from [-1, 1] to [lb, ub]
        """
        return self.scaler.inverse_transform(x)

class AbstractBenchmark(ABC):
    def __init__(self, dim):
        self.dim   = dim
        self.space = DesignSpace().parse([{'name' : f'x{i}', 'type' : 'num', 'lb' : -1, 'ub' : 1} for i in range(dim)])
    
    @abstractmethod
    def __call__(self, para : pd.DataFrame) -> np.ndarray:
        pass

class WhiteNoise(AbstractBenchmark):
    def __init__(self, dim):
        """
        eff_dim = 2
        """
        super().__init__(dim)
        self.eff_dim = 2
    
    def __call__(self, _params : pd.DataFrame) -> np.ndarray:
        return np.random.randn(_params.shape[0], 1)

class SynHDBench(AbstractBenchmark):
    def __init__(self, dim):
        """
        eff_dim = 2
        """
        super().__init__(dim)
        self.eff_dim = 2
    
    def __call__(self, _params : pd.DataFrame) -> np.ndarray:
        eff_dim = self.eff_dim
        params  = _params * 3.5 # XXX: scale [-1, 1] to [-3.5, 3.5]
        return - (((params.values[:,0:eff_dim]).sum(axis = 1) - 0.1)**2 * np.abs( np.sin(params.values[:,0:eff_dim]).sum(axis = 1)/(np.cos(params.values[:,0:eff_dim]+2).sum(axis = 1)+ 3) )).reshape(-1, 1)

class PymooDummy(AbstractBenchmark):
    def __init__(self, dim, prob, bounds : tuple = None):
        super().__init__(dim)
        self.prob    = prob
        assert self.prob.n_var <= self.dim, f"Effective dim {self.prob.n_var} less than actual dim {dim}"
        self.eff_dim = prob.n_var
        lb, ub       = prob.bounds() if bounds is None else bounds

        self.lb = -1 * np.ones(dim)
        self.ub = np.ones(dim)
        self.lb[:self.eff_dim] = lb
        self.ub[:self.eff_dim] = ub
        self.scaler = BenchParamScaler(self.lb, self.ub)
        self.best_y = self.prob.ideal_point()
        self.pareto_front = self.prob.pareto_front() - self.prob.ideal_point()
        if not np.isfinite(self.best_y).all():
            self.best_y = 0.
    
    def __call__(self, para : pd.DataFrame) -> np.ndarray:
        x = self.scaler.inverse_transform(para[self.space.numeric_names].values)[:, :self.eff_dim]
        return self.prob.evaluate(x) - self.best_y

class BraninDummy(PymooDummy):
    def __init__(self, dim = 25):
        super().__init__(dim, prob = get_problem('go-branin01'))

class BraninNoisy(BraninDummy):
    """
    The first two dimensions are relavent, the rest dimensions generate noisy
    corruption, the higher the dimension is, the more noisy the output would be
    """
    def __init__(self, dim = 25, factor = 0.1):
        super().__init__(dim)
        self.factor = factor

    def __call__(self, para : pd.DataFrame) -> np.ndarray:
        ret   = super().__call__(para)
        noise = np.random.randn(para.shape[0], self.dim - self.eff_dim).sum(axis = 1, keepdims = True)
        return ret + self.factor * noise

class RosenbrockDummy(PymooDummy):
    def __init__(self, dim = 100):
        super().__init__(dim, prob = get_problem("rosenbrock", n_var=2))

class Hartmann6Dummy(PymooDummy):
    def __init__(self, dim = 100):
        super().__init__(dim, prob = get_problem("go-hartmann6"))

class StyblinskiTangDummy(PymooDummy):
    def __init__(self, dim = 100):
        super().__init__(dim, prob = get_problem("go-styblinskitang"))

class Rosenbrock(PymooDummy):
    def __init__(self, dim = 100):
        super().__init__(dim, prob = get_problem("rosenbrock", n_var=dim))

class Ackley(PymooDummy):
    def __init__(self, dim = 100):
        super().__init__(dim, prob = get_problem("ackley", n_var=dim, a=20, b=1/5, c=2 * np.pi))

class AckleyDummy(PymooDummy):
    def __init__(self, dim = 100, eff_dim = 2):
        bounds = (-5 * np.ones(eff_dim), 10 * np.ones(eff_dim))
        super().__init__(dim, prob = get_problem("ackley", n_var=eff_dim, a=20, b=1/5, c=2 * np.pi), bounds = bounds)


class AckleyOffsetRotation(PymooDummy):
    def __init__(self, dim = 100, rank = None):
        super().__init__(dim, prob = get_problem("ackley", n_var=dim, a=20, b=1/5, c=2 * np.pi))
        self.offset  = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)
        if rank is None:
            rank = dim
        r       = np.random.randn(dim, rank)
        self.r  = np.matmul(r, r.T)
        self.r *= np.random.randn(dim)
    
    def __call__(self, para):
        x = self.scaler.inverse_transform(para[self.space.numeric_names]) - self.offset
        h = np.matmul(x, self.r)
        return self.prob.evaluate(h)

class AckleyCompressed(AbstractBenchmark):
    def __init__(self, dim = 100, eff_dim = 2, seed : int = None):
        super().__init__(dim)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.eff_dim = eff_dim
        self.linear  = nn.Linear(dim, eff_dim, bias = False)
        self.ackley  = Ackley(eff_dim)
        self.offset  = 2 * np.random.rand(self.dim) - 1 # [-1, 1]
    
    def __call__(self, para):
        with torch.no_grad():
            x    = para[self.space.numeric_names].values - self.offset
            h    = self.linear(torch.FloatTensor(x)).numpy()
            df_h = pd.DataFrame(h, columns = self.ackley.space.numeric_names)
            return self.ackley(df_h)

class Schwefel_12(AbstractBenchmark):
    """
    Used in " High Dimensional Bayesian Optimization Using Dropout"
    Suggested dimension: [5, 10, 20, 30]
    """
    def __init__(self, dim = 30):
        super().__init__(dim)
        self.best_y = self._raw_eval(np.zeros((1, dim))).squeeze()

    def _raw_eval(self, x : np.array) -> np.ndarray:
        obj = np.zeros((x.shape[0], 1))
        for i in range(self.dim):
            item  = (x[:, :(i+1)]**2).sum(axis = 1, keepdims = True)
            obj  += item
        return obj

    def __call__(self, para) -> np.ndarray:
        return self._raw_eval(para[self.space.numeric_names].values) - self.best_y

if __name__ == '__main__':
    prob = Schwefel_12(3)
    x    = prob.space.sample(10)
    y    = prob(x)
    print(y)
    print('ok')
