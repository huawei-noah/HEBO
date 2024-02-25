# MCBO: Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization

This library provides an easy way to mix-and-match Bayesian optimization components in order to run new
and existing mixed-variable or combinatorial Bayesian optimization. Motivations and principles are described in
[this paper](https://arxiv.org/pdf/2306.09803.pdf).

<p align="center">
<img src="paper_results/images/all_mix_match.PNG" width="750"/>
</p>

**Disclaimer:** the figure above shows that this library allows to build and run BO
algorithms made of the same primitives as several published BO methods, but
they should not be considered as their official implementations.

## Installation of MCBO

Tested on Ubuntu 18.04 and python 3.8.

Create a conda environment and activate it.

```shell
conda create -n mcbo_env python=3.8
conda activate mcbo_env

# install MCBO to be able to run optimization on custom problems
pip install -e .

# You can specify the path where all results will be stored by default
echo './results/' > ./mcbo/tasks/eda_seq_opt/results_root_path.txt

# if you want to run optimization of built-in Black-box, run the following (takes a while to get everything installed)
chmod u+x ./bbox_setup.sh
./bbox_setup.sh
```

## Implemented Tasks

### Synthetic

- 21 SFU test functions
- `ackley`: Ackley 20 caterorical dimensions (11 categories each).
- `ackley-53`: Ackley 50 binary dimensions, 3 numerical.
- `pest`: Pest Control

### Real-world

- `antibody_design`: Antibody Design
- `rna_inverse_fold`: RNA Inverse Folding
- `aig_optimization`: EDA Sequence Optimization (AIG sequence optimization)
- `aig_optimization_hyp`: EDA Sequence and Parameter Optimization (AIG sequence and parameter optimization)
- `mig_optimization`: MIG Sequence Optimization
- `svm_opt`: SVM hyperparameter tuning and feature selection
- `xgboost_opt`: XG-Boost hyperparameter tuning

### How to access a task

All the tasks are accessible via the `task_factory` function. Below we show how to obtain the `task`  and `search_space`
class for the RNA inverse fold task.

```
from mcbo import task_factory
import torch

task_name = 'rna_inverse_fold'
task = task_factory(task_name=task_name)
search_space = task.get_search_space()
```

## Implemented Primitives

### Surrogate model

- `gp_o`: GP with overlap kernel.
- `gp_to`: GP with transformed-overlap kernel.
- `gp_hed`: GP with the Hamming embedding via dictionary kernel.
- `gp_ssk`: GP with string subsequence kernel.
- `gp_diff`: GP with diffusion kernel.
- `gp_rd`:  GP with random tree decomposition additive kernel (for very high dimension).
- `lr_sparse_hs`: Bayesian linear regression with Hoorseshoe prior.

### Acquisition function optimizers

- `ga`: Genetic algorithm.
- `sa`: Simulated Annealing.
- `ls`: Exhaustive Local Search.
- `is`: Interleaved search with Hill-Climbing and Gradient-Descent.
- `mab`: Multi-Armed Bandit for categorical and Gradient-Descent for numerical.
- `mp`: Message passing (compatible with `gp_rd` model)

### Acquisition functions

- `ei`: Expected Improvement.
- `lcb`: Lower Confidence Bound.
- `ts`: Thompson Sampling.

### Trust-region

- `none`: No trust region.
- `basic`: Hamming distance for categorical variables, hyperectangle limit for numerical variables.

### Non-BO baselines

- `rs`: Random Search.
- `hc`: Hill Climbing.
- `sa`: Simulated Annealing.
- `ga`: Genetic Algorithm.
- `mab`: Multi-Armed Bandit.

## Simple optimization example

**Remark:** in our library, it is always assumed that the black-box optimization is a **minimization** problem.

- A simple script to build Casmopolitan optimizer and run it on RNA inverse fold.

```python
import torch
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder

if __name__ == '__main__':
    task_kws = dict(target=65)
    task = task_factory(task_name='rna_inverse_fold', **task_kws)
    search_space = task.get_search_space()
    bo_builder = BoBuilder(
        model_id='gp_to', acq_opt_id='is', acq_func_id='ei', tr_id='basic', init_sampling_strategy="sobol_scramble"
    )

    optimizer = bo_builder.build_bo(search_space=search_space, n_init=20, device=torch.device("cuda"))

    for i in range(100):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i + 1:3d}/{100:3d} - f(x) = {y[0, 0]:.3f} - f(x*) = {optimizer.best_y.item():.3f}')

    # Access history of suggested points and black-box values
    all_x = search_space.inverse_transform(optimizer.data_buffer.x)
    all_y = optimizer.data_buffer.y.cpu().numpy()
```

- To run and save optimization results with several seeds and optimizers,
  we provide the `./experiments/run_task_exps.py` script calling `run_experiment` function.

```shell
tr_id="basic"                    # can be 'none'
init_sampling_strategy="uniform" # can be "sobol" or "sobol_scramble" to use initial sobol samples

python ./experiments/run_task_exps.py --device_id 0 --task_id "rna_inverse_fold" \
  --optimizers_ids gp_to__is__ei__${tr_id}__${init_sampling_strategy} gp_hed__is__ei__${tr_id}__${init_sampling_strategy} \
  --seeds 42 43 --verbose 2

```

## How to use MCBO?

In [./tutorials](./tutorials/) folder, we provide several step-by-step indications on how to use and extend the library,
including guidance to:

- run an MCBO on a custom task
- add MCBO component such as a new acquisition function or a new surrogate model
- run a new method on set of benchmarks
- generate result plots to visualize the evolution of the regret.

## Reproduce the results of the paper.

#### Running

It is possible to reproduce our optimization results by running the script [./all_runs.sh](./experiments/all_runs.sh).

```shell
chmod u+x ./experiments/all_runs.sh
./experiments/all_runs.sh
```

Results will be saved in `./resutls/` or in user specified path (modify `RESULTS_DIR`
in [global_settings.py](./mcbo/global_settings.py)).

#### Visualize

```bash
unzip ./paper_results/data.zip
```

We provide [notebooks](./paper_results) to visualize rankings and regrets. Many plotting tools are available
in [general_plot_utils.py](./mcbo/utils/general_plot_utils.py).

## Library Roadmap

#### Features
- [x] Allows restart from checkpoints
- [x] Add random tree-based additive GP kernel as surrogate model
- [x] Add message-passing acquisition function optimizer
- [x] Add an option to use Sobol sampling instead of uniform sampling of the initial points suggested by BO.   
- [ ] Support multi-objective acquisition functions such as MACE.
- [ ] Add sparse-GP surrogates.
- [ ] Add NP-based surrogates.
- [ ] Build a comprehensive documentation
- [ ] Add non-myopic acquisition functions (KG, MES).
- [ ] Handle black-box constraints.
- [ ] Handle multi-fidelity MCBO.
- [ ] Implement probabilistic reparameterization for acquisition function optimization
- [ ] Support optimizing in a table of points instead of in the full search space.

#### Debug
- [] improve the way we cope with hallucinatory points (to prevent sampled values from exploding)



We invite you to contribute to the development of the MCBO library notably by proposing implementation of new modules,
providing new tasks or spotting issues.

## Cite us

If you use this library, please cite [MCBO paper](https://arxiv.org/pdf/2306.09803.pdf):
> Kamil Dreczkowski, Antoine Grosnit, and Haitham Bou Ammar. Framework and Benchmarks for Combinatorial and
> Mixed-variable Bayesian Optimization

###### bibtex

```bibtex 
@misc{dreczkowski2023mcbo,
      title={Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization}, 
      author={Kamil Dreczkowski and Antoine Grosnit and Haitham Bou Ammar},
      year={2023},
      eprint={2306.09803},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

For official implementations of the existing algorithms that our library can build and which inspired
some of our primitives, we refer to

- BOSS: https://github.com/henrymoss/BOSS
- COMBO: https://github.com/QUVA-Lab/COMBO/
- Casmopolitan: https://github.com/xingchenwan/Casmopolitan
- CoCaBO: https://github.com/rubinxin/CoCaBO_code
- BOCS: https://github.com/baptistar/BOCS
- BOiLS: https://github.com/huawei-noah/HEBO/tree/master/BOiLS
- BODi: https://github.com/aryandeshwal/BODi/
- RDUCB: https://github.com/huawei-noah/HEBO/tree/master/RDUCB
