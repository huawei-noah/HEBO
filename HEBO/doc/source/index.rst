.. BO documentation master file, created by
   sphinx-quickstart on Wed Nov 11 15:20:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/huawei-noah/HEBO/tree/master/HEBO

HEBO
==============================


Heteroscedastic evolutionary bayesian optimisation, developed by Huawei Noah's Ark Lab

Quick start
--------------

.. code-block:: python

    import pandas as pd
    import numpy  as np

    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    def obj(params : pd.DataFrame) -> np.ndarray:
        return ((params.values - 0.37)**2).sum(axis = 1).reshape(-1, 1)
            
    space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : -3, 'ub' : 3}])
    opt   = HEBO(space)
    for i in range(5):
        rec = opt.suggest(n_suggestions = 4)
        opt.observe(rec, obj(rec))
        print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))

Tuning sklearn estimators
----------------------------

.. code-block:: python

    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    from hebo.sklearn_tuner import sklearn_tuner

    space_cfg = [
        {'name' : 'max_depth', 'type' : 'int', 'lb' : 1, 'ub' : 20},
        {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1e-4, 'ub' : 0.5},
        {'name' : 'max_features', 'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
        {'name' : 'bootstrap', 'type' : 'bool'},
        {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
        ]
    X, y   = load_boston(return_X_y = True)
    result = sklearn_tuner(RandomForestRegressor, space_cfg, X, y, metric = r2_score, max_iter = 16)


Features
******************

- Continuous and categorical design parameters
- Support creating new parameter type
- Modular and flexible BO building blocks
- Multiple probabilistic models including GP, RF and BNN 
- Ready to use optimizers:
    - Constrained and multi-objective optimisation
    - Contextual optimisation

.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Tutorial

   installation
   design_space.ipynb
   optimisation
   sklearn_tuner

   model_cmp
   mo_constrained.ipynb
   alebo_demo.ipynb
   pymoo_evolution
   custom
   sgmcmc

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
