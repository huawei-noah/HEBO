# TODO

- Documentation
- Options for MACE
    - Power transformation
    - Noise exploration
    - Verbose
- EI, UCB, PI, MES, TS
- MO model wrapper
- Constrained and multi-objective opt
- General single/multi-objective constrained optimizer
- Gaussian process:
    - Support different x/y scalers
    - Support different kernels (Mat32, Mat52, RBF)
    - Support different categorical processing methods
- Build wheel, setup requirement

# Done

- Upgrade to pymoo-0.4.2
- Optimizer API
- Design space redesign
- Contextual BO API in `suggest`
- Options for MACE:
    - Model
    - Num random

# Features to support

- Basic Bayesian optimisation
- Mainstream acquisition functions
- Constrained Bayesian optimisation (weighted EI, Thompson sampling, entropy search) 
- Multi-objective Bayesian optimisation
- Multi-fidelity Bayesian optimisation
- Safe Bayesian optimisation
- High-dimensional Bayesian optimisation
- Batch optimisation
- Meta learning (learn from past optimisation tasks or relavent data)
- Prior embedding
- Support other probablistic models:
    - BNN via variational inference
    - BNN via SGDMCMC
    - Deep ensemble
