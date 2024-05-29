import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


if __name__ == "__main__":
    def obj(params: pd.DataFrame) -> np.ndarray:
        return ((params.values - 0.37) ** 2).sum(axis=1).reshape(-1, 1)


    space = DesignSpace().parse([{'name': f'x{i}', 'type': 'num', 'lb': -3, 'ub': 3} for i in range(5)])
    opt = HEBO(space=space, input_constraints=["x0 - x1", lambda x: x["x3"] - x["x4"]])
    for i in range(5):
        rec = opt.suggest(n_suggestions=4)
        opt.observe(rec, obj(rec))
        print(rec)
        print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))