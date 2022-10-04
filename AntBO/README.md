# AntBO 

**A combinatorial Bayesian Optimisation framework enabling efficient in
silico design of the CDRH3 region.**

![AntBO overview](./figures/AntBO_illustrationPNG.PNG?raw=true)

This repo provides the official implementation of _AntBO_,
as well as all the code needed to reproduce the experiments presented in
[AntBO: Towards Real-World Automated Antibody Design with Combinatorial Bayesian Optimisation](https://arxiv.org/abs/2201.12570).


## Setup
The code has been tested on  `Ubuntu 18.04.4 LTS` with `python 3.9.7` 

#### Conda environment

```bash
conda env create -f environment.yaml 
conda activate DGM
```

#### Antigen simulator: _Absolut!_

Install [Absolut!](https://github.com/csi-greifflab/Absolut) by following instructions 
of their README: https://github.com/csi-greifflab/Absolut#installation. 

#### Install all pre-computed structures (~33GB)

```bash
PATH_TO_ABSOLUT=./Absolut # put the path of the directory where Absolut! is installed
cd $PATH_TO_ABSOLUT
aria2c -i ../urls.txt --auto-file-renaming=false --continue=true
```

## Demo

To run AntBO (as well as the other baselines COMBO, Genetic Algorithm, 
and Random Search), the user needs to specify a `config.yaml` file containing the parameters of the run ([a default 
config file](./bo/config.yaml) is also provided).

Given a config file, AntBO can be run using the following command line:
```shell
 python ./bo/main.py --config ./bo/config.yaml --n_trials 5 --seed 42 --antigens_file ./dataloader/core_antigens.txt 
```
- Using `--n_trials 5` and `--seed 42` implies that the optimisation will be run over 5 random seeds starting from 
seed 42.
- `--antigens_file ./dataloader/core_antigens.txt` indicates that **AntBO** will be run on each of
 the antigens listed in the file [./dataloader/core_antigens.txt](./dataloader/core_antigens.txt).
- The results will be stored in the folder specified in the `config.yaml` file (under the `save_dir` field).

#### Results visualisation
To plot the regret curve associated to one run of **AntBO** on an antigen, one can either run the following:

```python
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from utilities.misc_utils import load_w_pickle
from typing import *

from bo.main import BOExperiments
from task.utils import plot_mean_std

# --- Select for which antigen you want to plot the regret curve --- #
antigen_name = ...

# --- These must match your config.yaml --- #
save_path: str = ... 
kernel_type = "transformed_overlap"  
cdr_constraints = 1
seq_len: int = 11
search_strategy = "local"

init_seed = 42
n_trials = 5

# ----------- Collect evolution of scores for each seed ----------- #

results = []
for seed in range(init_seed, init_seed + n_trials):
    
    result_path_root = BOExperiments.get_path(
        save_path=save_path,
        antigen=antigen_name,
        kernel_type=kernel_type, 
        seed=seed,
        cdr_constraints=cdr_constraints,
        seq_len=seq_len,
        search_strategy=search_strategy,
    )

    result_path = os.path.join(result_path_root, 'results.csv')

    if not os.path.exists(result_path):
        continue

    data = pd.read_csv(result_path).BestValue.values
    
    results.append(data)

# --------------------- Plot the regret curve --------------------- #

ax = plot_mean_std(results)
ax.set_xlabel("Num. evaluations", fontsize=16)
```

or rely on the [convergence curve plotting script](./visualise_results/plot_convergence_curve.py) fed with a 
[configuration file](visualise_results/convergence_curve_config.yaml).

```shell 
python ./visualise_results/plot_convergence_curve.py --config ./visualise_results/convergence_curve_config.yaml
```

## Absolut 3D Visualisation

Follow the instructions in visualise3d.txt




## Cite us
```bibtex
@misc{https://doi.org/10.48550/arxiv.2201.12570,
  doi = {10.48550/ARXIV.2201.12570},
  url = {https://arxiv.org/abs/2201.12570},
  author = {Khan, Asif and Cowen-Rivers, Alexander I. and Deik, Derrick-Goh-Xin and Grosnit, Antoine and Dreczkowski, Kamil and Robert, Philippe A. and Greiff, Victor and Tutunov, Rasul and Bou-Ammar, Dany and Wang, Jun and Bou-Ammar, Haitham},
  keywords = {Biomolecules (q-bio.BM), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), Machine Learning (stat.ML), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {AntBO: Towards Real-World Automated Antibody Design with Combinatorial Bayesian Optimisation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Contributors
**Asif Khan**, **Alexander I. Cowen-Rivers**, **Derrick-Goh-Xin Deik**, **Antoine Grosnit**,
**Kamil Dreczkowski**, **Philippe A. Robert**, **Victor Greiff**,
**Rasul Tutunov**, **Dany Bou-Ammar**, **Jun Wang**, **Haitham Bou-Ammar**
-- _Huawei Noah's Ark lab_.
