![](hebo.png)
```
pip install HEBO
```
# README

Bayesian optimsation library developped by Huawei Noahs Ark Decision Making and Reasoning (DMnR) lab. The <strong> winning submission </strong> to the [NeurIPS 2020 Black-Box Optimisation Challenge](https://bbochallenge.com/leaderboard). 

Summary             |  Ablation
:-------------------------:|:-------------------------:
[Results](https://github.com/huawei-noah/HEBO/blob/master/HEBO/summary_plot2.pdf) | [Results](https://github.com/huawei-noah/HEBO/blob/master/HEBO/summary_ablation2.pdf)

# Contributors 

<strong> Alexander I. Cowen-Rivers, Wenlong Lyu, Rasul Tutunov, Zhi Wang, Antoine Grosnit, Ryan Rhys Griffiths, Alexandre Max Maraval, Hao Jianye, Jun Wang, Jan Peters, Haitham Bou Ammar </strong>

## Installation

You can install HEBO from pypi by

```bash
pip install HEBO
```

You can also build from source code to obtain our latest update:

```bash
cd HEBO
pip install -e .
```


## Documentation

Online documentation can be seen [here](https://hebo.readthedocs.io/en/latest/)

You can also build the documentation by your self

```bash
pip install -r dev-requirements.txt
cd doc
make html
```

## Demo

```python
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
```

## Auto Tuning via Sklearn Estimator

```python
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
```

## Test

```bash
pytest -v test/ --cov ./bo --cov-report term-missing --cov-config ./test/.coveragerc
```

## Reproduce Experimental Results

- See `archived_submissions/hebo`, which is the exact submission that won the NeurIPS2020 Black-Box Optimsation Challenge.
- Use `run_local.sh` in [bbo_challenge_starter_kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit/) to reproduce `bayesmark` experiments, you can just drop `archived_submissions/hebo` to the `example_submissions` directory.
- The `MACEBO` in `bo.optimizers.mace` is the same optimiser, with same hyperparameters but a modified interface (bayesmark dependency removed).


## Features

- Continuous, integer and categorical design parameters.
- Constrained and multi-objective optimsation.
- Contextual optimsation.
- Multiple surrogate models including GP, RF and BNN.
- Modular and flexible Bayesian Optimisation building blocks.


## Cite Us

Cowen-Rivers, Alexander I., et al. "An Empirical Study of Assumptions in Bayesian Optimisation." arXiv preprint arXiv:2012.03826 (2021).

## BibTex
```
@article{Cowen-Rivers2022-HEBO,
author = {Cowen-Rivers, Alexander and Lyu, Wenlong and Tutunov, Rasul and Wang, Zhi and Grosnit, Antoine and Griffiths, Ryan-Rhys and Maravel, Alexandre and Hao, Jianye and Wang, Jun and Peters, Jan and Bou Ammar, Haitham},
year = {2022},
month = {07},
pages = {},
title = {HEBO: Pushing The Limits of Sample-Efficient Hyperparameter Optimisation},
volume = {74},
journal = {Journal of Artificial Intelligence Research}
}
```
