# Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief

Code to reproduce the experiments in [Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief](https://nips.cc/Conferences/2022/Schedule?showEvent=54842).

## Installation
1. Install [MuJoCo 2.0.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco200`.
2. Create a conda environment and install requirements.
```
cd PMDB
conda env create -f PMDB_env.yml
conda activate PMDB_env
```

## Usage
For example, use the following command to run Hopper-medium-v2 benchmark in D4RL.

```
python main.py --task=hopper-medium-v2
```
Detailed configuration can be found in `config.py`.


#### Logging
By default, TensorBoard logs are generated in the `log/` directory.


## Citing PMDB

```
@inproceedings{guo2022pmdb,
  title={Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief},
  author={Kaiyang Guo and Yunfeng Shao and Yanhui Geng},
  booktitle{NeurIPS},
  year={2022}
}
```
