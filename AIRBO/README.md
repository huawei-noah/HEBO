# AIRBO: Efficient Robust Bayesian Optimization for Arbitrary Uncertain Inputs

<div style="text-align:center"><img src="./figures/logo.png" alt="drawing" width="300"/>

Welcome to the repository for our paper: [Efficient Robust Bayesian Optimization for Arbitrary Uncertain Inputs, NeurIPS 2023](https://arxiv.org/abs/2310.20145). 

## Overview

Bayesian optimization is a powerful framework for global optimization of expensive black-box functions. However, most existing methods assume that the inputs are perfectly known and fixed, which is often not the case in real-world applications. In AIRBO, we propose a novel method for efficient robust Bayesian optimization that can handle arbitrary input uncertainties. In particular, we design an MMD-based kernel to measure uncertain inputs in an RKHS and employ Nystrom approximation to boost the inference of GP posterior.

<div style="text-align:center"><img src=".\figures\AIRBO_neurips_poster.png" alt="drawing" width="1000"/>



## Features

- Can take random variables (*e.g.*, input with noise)  as inputs and propagate the input uncertainty to the posterior.
- Can be applied to arbitrary input distribution if we can sample from it.
- Aim to find a robust optimum whose expected function value is best under the input uncertainty. 
- Efficient posterior inference (40 times more efficient than the integral kernel).


## Requirements

- Python 3.8 or higher

- Botorch 0.9.2 or higher

- GPytorch 1.11 or higher

**see more package dependencies in the requirements.txt*

## Usage

To use our implementation, follow these steps:

1. Clone the repository

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. To compare the modeling performance of robust surrogate models, run:

    ```bash
    sh scripts/compare_surrogate_models.sh
    ```
    
4. To compare the optimization on benchmark functions:

    ```bash
    sh scripts/compare_robust_optimization.sh
    ```
    
5. Run robust optimization in a push-world game (need Box2D and pygame packages)
    ```bash
    sh scripts/compare_optimization_in_push_world.sh
    ```

## Citation

If you find this work useful, please cite our paper: Yang, Lin, et al. "Efficient Robust Bayesian Optimization for Arbitrary Uncertain Inputs."  NeurIPS 2023.

```latex
@inproceedings{yang2023efficient,
  title={Efficient Robust Bayesian Optimization for Arbitrary Uncertain inputs},
  author={Yang, Lin and Lyu, Junlong and Lyu, Wenlong and Chen, Zhitang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

