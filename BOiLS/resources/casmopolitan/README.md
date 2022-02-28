## Code for 'Think Glocal and Act Local: Bayesian Optimisation for Categorical and Mixed Search Spaces'

This repo contains the current implementation of the CASMOPOLITAN algorithm. 

[Link to paper](https://arxiv.org/abs/2102.07188)

If you find our paper or this repo to be useful for your research, please consider citing:
```
@article{wan2021think,
  title={Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces},
  author={Wan, Xingchen and Nguyen, Vu and Ha, Huong and Ru, Binxin and Lu, Cong and Osborne, Michael A},
  journal={International Conference on Machine Learning (ICML) 38},
  year={2021}
}
```

# Dependencies
```
Anaconda Python 3.6
gpytorch==0.3.5
pytorch==1.6.0
numpy==1.16.4
pandas==0.25.1
tqdm (for progress bar visualisation)==4.32.1
xg-boost==0.90 (to run XGBoost on MNIST task)
scikit-learn==0.21.2
```

# Examples to reproduce results in paper
```bash
    python3 main.py -p pest
    python3 main.py -p func2C --max_iters 200
    python3 main.py -p ackley53 --max_iters 400
```
where ```-p``` specifies the problem, ```-n``` specifies the number of trust region to initialise. Look at ```main.py``` to see other settings that can be tuned. ```-a``` specifies the acquisition function.

# To run a new custom problem
1. Go to ```./test_funcs/base.py``` for the implementation of the base class. A new problem should be a derived class
   that properly implements all the required methods of the base class. Note that by default the new problem has property
   ```problem_type == 'categorical'```. If your new problem is in the mixed space, you need to change this to ```mixed``` (See examples
   in ```./mixed_test_func```).
2. Edit the appropriate imports and other specifications and run from ```main.py``` as usual.    


# Acknowledgements

This code repository uses materials from the following public repositories. The authors thank the respective repository maintainers

1. TuRBO: Eriksson, D., Pearce, M., Gardner, J. R., Turner, R., & Poloczek, M. (2019). Scalable global optimization via local Bayesian optimization. Advances in Neural Information Processing Systems, 32 (NeurIPS).
   Code repo: https://github.com/uber-research/TuRBO/
2. CoCaBO: Ru, B., Alvi, A. S., Nguyen, V., Osborne, M. A., & Roberts, S. J. (2020). Bayesian Optimisation over Multiple Continuous and Categorical Inputs. International Conference on Machine Learning, 37 (ICML).
   Code repo: https://github.com/rubinxin/CoCaBO_code
3. COMBO: Oh, C., Tomczak, J. M., Gavves, E., & Welling, M. (2019). Combinatorial Bayesian Optimization using the Graph Cartesian Product. Advances in Neural Information Processing Systems, 32 (NeurIPS).
Code repo: https://github.com/QUVA-Lab/COMBO
4. Starting kit for NeurIPS 2020 Black-Box Optimisation Challenge. Code repo: https://github.com/rdturnermtl/bbo_challenge_starter_kit
5. Bliek, L., Guijt, A., Verwer, S., & de Weerdt, M. (2020). Black-box mixed-variable optimisation using a surrogate model that satisfies integer constraints. arXiv preprint arXiv:2006.04508. Code repo: https://github.com/lbliek/MVRSM
