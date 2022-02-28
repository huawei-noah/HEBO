ABC\_RL
--------
Reinforcement learning for logic synthesis.

This is the source codes for our paper "Exploring Logic Optimizations with Reinforcement Learning and Graph Convolutional Network", published at 2nd ACM/IEEE Workshop on Machine Learning for CAD (MLCAD), Nov. 2020.

The authors include [Keren Zhu](https://krz.engineer), Mingjie Liu, Hao Chen, Zheng Zhao and David Z. Pan.

--------
# Prerequsites

# Python environment

The project is tested in the environment:
```
Python 3.7.5
Pytorch 1.3
```

The project has other dependencies such as `numpy, six, etc.`
Please installing the dependencies correspondingly.

# abc\_py

The project requires the Python API, [abc\_py](https://github.com/krzhu/abc\_py), for [Berkeley-abc](https://github.com/berkeley-abc/abc).

Please refer to the Github page of abc\_py for installing instruction.

--------

# Benchmarks

Benmarks can be found in [url](https://ddd.fit.cvut.cz/prj/Benchmarks/index.php?page=download).

--------

# Usage

The current version can execute on combinational `.aig` and `.blif` benchmarks.
To run the REINFORCE algorithm, please first edit the `python/rl/testReinforce.py` for the benchmark circuit.
And execute `python3 testReinforce.py`


--------

# Contact

Keren Zhu, UT Austin (keren.zhu AT utexas.edu)
