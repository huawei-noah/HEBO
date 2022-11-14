# Saut\'e and Simmer {RL}: Safe Reinforcement Learning Using Safety State Augmentation

###  Sauté RL: Almost Surely Safe RL Using State Augmentation

Satisfying safety constraints almost surely (or with probability one) can be critical for deployment of Reinforcement
Learning (RL) in real-life applications. For example, plane landing and take-off should ideally occur with probability
one. We address the problem by introducing Safety Augmented (Saute) Markov Decision Processes (MDPs), where the safety
constraints are eliminated by augmenting them into the state-space and reshaping the objective. We show that Saute MDP
satisfies the Bellman equation and moves us closer to solving Safe RL with constraints satisfied almost surely. We argue
that Saute MDP allows to view Safe RL problem from a different perspective enabling new features. For instance, our
approach has a plug-and-play nature, i.e., any RL algorithm can be "sauteed". Additionally, state augmentation allows
for policy generalization across safety constraints. We finally show that Saute RL algorithms can outperform their
state-of-the-art counterparts when constraint satisfaction is of high importance.



### Effects of Safety State Augmentation on Safe Exploration
Safe exploration is a challenging and important problem in model-free reinforcement learning (RL). Often the safety cost
 is sparse and unknown, which unavoidably leads to constraint violations -- a phenomenon ideally to be avoided in 
 safety-critical applications. We tackle this problem by augmenting the state-space with a safety state, which is 
 nonnegative if and only if the constraint is satisfied. The value of this state also serves as a distance toward 
 constraint violation, while its initial value indicates the available safety budget. This idea allows us to derive 
 policies for scheduling the safety budget during training. We call our approach Simmer (Safe policy IMproveMEnt for 
 RL) to reflect the careful nature of these schedules. We apply this idea to two safe RL problems: RL with constraints 
 imposed on an average cost, and RL with constraints imposed on a cost with probability one. Our experiments suggest 
 that simmering a safe algorithm can improve safety during training for both settings. We further show that Simmer can
  stabilize training and improve the performance of safe RL with average constraints.


## Installation 

The following installation commands were tested in Ubuntu 18.04.

Create a conda environment 

```console
conda config --append channels conda-forge
conda env create -f sauterl.yml
conda activate sauterl
```

Our implementation is based on the Open AI safety starter agents. To install the Open AI libraries run the following commands:

```console
mkdir safe-rl
cd safe-rl
git clone https://github.com/openai/safety-starter-agents.git && cd safety-starter-agents && pip install -e . && cd ..
git clone https://github.com/openai/safety-gym.git  && cd safety-gym && pip install -e . && cd ..
cd .. 
```

Install the remaining libraries 

```console
pip install -r pip_repos.txt
``` 

## Saute&#769;ing an environment
Using our approach in practice is straightforward and requires just three steps: 

1. Creating a safe environment 
2. Saute&#769;ing the safe environment 
3. Running a standard Reinforcement Learning algorithm 

### Creating a safe environment

In order to create a custom safe environment we need to define the safety cost and to inherit the rest of the definitions from the standard gym environment `MyEnv` 
and the provided class `SafeEnv`. 


```python 
from envs.common.safe_env import SafeEnv

class MySafeEnv(SafeEnv, MyEnv):
    """New safety class"""
    def _safety_cost_fn(self, state:np.ndarray, action:np.ndarray, next_state:np.ndarray) -> np.ndarray:
        """Define the safety cost here."
```

The class `SafeEnv` contains the changes to the `step` method, which incorporates the safety constraints. Note that we assume that there is a method `MyEnv._get_obs()`.

### Saute&#769;ing a safe environment 
Safety state augmentation (saute&#769;ing) is done in a straightforward manner. Assume a safe environment is defined in a class `MySafeEnv`. The saute&#769;ed environment is defined using a decorator `saute_env`, which contains all the required definitions. Custom and overloaded functions can be defined in the class body. 

```python
from envs.common.saute_env import saute_env

@saute_env
class MySautedEnv(MySafeEnv):
    """New sauteed class."""    
```

## Running 

We release a few tested safe environments, which can be evaluated using main.py. The file takes two arguments: the experiment identifier and the number of experiments for a particular algorithm to run in parallel. For instance, 

```console 
python main.py --experiment 11 --num-exps 5
```


Our experiments:

ID | Environment | Algorithms | Type of Experiment 
--- | --- | --- | ---
10 | Pendulum swing-up | SAC, Langrangian SAC, Saute&#769; SAC | Performance 
11 | Pendulum swing-up | PPO, Langrangian PPO, Saute&#769; PPO | Performance 
12 | Pendulum swing-up | TRPO, Langrangian TRPO, Saute&#769; TRPO, CPO | Performance 
13 | Pendulum swing-up | Saute&#769; TRPO | Ablation 
14 | Pendulum swing-up | Simmer TRPO | PI controlled schedule
15 | Pendulum swing-up | Simmer TRPO | Q learning controlled schedule
20 | Double Pendulum | TRPO, Saute&#769; TRPO, CPO| Performance
21 | Double Pendulum | Lagrangian TRPO | Performance
22 | Double Pendulum | Saute&#769; TRPO | Naive generalization across safety budgets 
23 | Double Pendulum | Saute&#769; TRPO | Smart generalization across safety budgets 
24 | Double Pendulum | Saute&#769; TRPO | Ablation over unsafe reward value 
30 | Reacher | TRPO, Langrangian TRPO, Saute&#769; TRPO, CPO | Performance
40 | Safety gym | TRPO, Langrangian TRPO, Saute&#769; TRPO, CPO | Performance

## Output 

By default the output is saved to `./logs/` directory (the directory can be modified in the method `set_all_overrides` in the `BaseRunner` class). 

By default no checkpoints are saved, but the results are tracked in the tensorboard.


## Citation

If you find our code useful please cite the following papers

```
@article{sootla2022saute,
  title={Saut\'e RL: Almost Surely Safe Reinforcement Learning Using State Augmentation},
  author={Sootla, Aivar and Cowen-Rivers, Alexander I and Jafferjee, Taher and Wang, Ziyan and Mguni, David and Wang, Jun and Bou-Ammar, Haitham},
  journal={arXiv preprint arXiv:2202.06558},
  year={2022}
}

@article{sootla2022simmer,
  title = {Effects of Safety State Augmentation on Safe Exploration},
  author = {Sootla, Aivar and Cowen-Rivers, Alexander I. and Wang, Jun and Bou-Ammar, Haitham},
  journal={arXiv preprint arXiv:2206.02675},
  year={2022}
}
```

or  

```
@misc{sootla_saute_2022_git,
	 title={Saut\'e and Simmer {RL}:  Safe Reinforcement Learning Using Safety State Augmentation}, 
   url = {https://github.com/huawei-noah/HEBO/tree/master/SIMMER},
   year = {2022},
	 author = {Sootla, Aivar and Cowen-Rivers, Alexander I. and Jafferjee, Taher and Wang, Ziyan},
}
```
