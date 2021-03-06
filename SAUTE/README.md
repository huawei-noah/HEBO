# Almost Surely Safe RL Using State Augmentation - Saute&#769; RL

## About 

Satisfying safety constraints almost surely (or with probability one) can be critical for deployment of Reinforcement Learning (RL) in real-life applications. For example, plane landing and take-off should ideally occur with probability one. We address the problem by introducing Safety Augmented (Saute&#769;) Markov Decision Processes (MDPs), where the safety constraints are eliminated by augmenting them into the state-space and reshaping the objective. We show that Saute&#769; MDP satisfies the Bellman equation and moves us closer to solving Safe RL with constraints satisfied almost surely. We argue that Saute&#769; MDP allows to view Safe RL problem from a different perspective enabling new features. For instance, our approach has a plug-and-play nature, i.e., any RL algorithm can be "saute&#769;ed". Additionally, state augmentation allows for policy generalization across safety constraints. In our experiments, we show that Saute&#769; RL algorithms outperforms their state-of-the-art counterparts when constraint satisfaction is of high importance. 

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
10 | Pendulum swing-up | SAC, Langrangian SAC, Saute SAC | Performance 
11 | Pendulum swing-up | PPO, Langrangian PPO, Saute PPO | Performance 
12 | Pendulum swing-up | TRPO, Langrangian TRPO, Saute TRPO, CPO | Performance 
13 | Pendulum swing-up | Saute TRPO | Ablation 
20 | Double Pendulum | TRPO, Saute TRPO, CPO| Performance
21 | Double Pendulum | Lagrangian TRPO | Performance
22 | Double Pendulum | Saute TRPO | Naive generalization across safety budgets 
23 | Double Pendulum | Saute TRPO | Smart generalization across safety budgets 
24 | Double Pendulum | Saute TRPO | Ablation over unsafe reward value 
30 | Reacher | TRPO, Langrangian TRPO, Saute TRPO, CPO | Performance
40 | Safety gym | TRPO, Langrangian TRPO, Saute TRPO, CPO | Performance


## Output 

By default the output is saved to `./logs/` directory (the directory can be modified in the method `set_all_overrides` in the `BaseRunner` class). 

By default no checkpoints are saved, but the results are tracked in the tensorboard.


## Citation

If you find our code useful please cite our paper!

```
@article{sootla2022saute,
  title={SAUT\'E RL: Almost Surely Safe Reinforcement Learning Using State Augmentation},
  author={Sootla, Aivar and Cowen-Rivers, Alexander I and Jafferjee, Taher and Wang, Ziyan and Mguni, David and Wang, Jun and Bou-Ammar, Haitham},
  journal={arXiv preprint arXiv:2202.06558},
  year={2022}
}
```
