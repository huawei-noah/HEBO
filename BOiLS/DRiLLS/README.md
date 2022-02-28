# DRiLLS <img align="right" width="10%" src="doc/img/drills-logo.png">
Deep Reinforcement Learning for Logic Synthesis Optimization

## Abstract
Logic synthesis requires extensive tuning of the synthesis optimization flow where the quality of results (QoR) depends on the sequence of opti-mizations used.  Efficient design space exploration ischallenging due to the exponential number of possible optimization  permutations. Therefore,  automating the optimization process is necessary. In this work, we propose a novel reinforcement learning-based methodology  that  navigates  the  optimization  space  without human intervention.  We demonstrate the training of an Advantage Actor Critic (A2C) agent that seeks to minimize area subject to a timing constraint.  Using the proposed framework, designs can be optimized autonomously with no-humans in-loop.

## Paper
DRiLLS has been presented at ASP-DAC 2020 and the manuscript is available on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9045559). A pre-print version is available on [arXiv](https://arxiv.org/abs/1911.04021).

## Setup
DRiLLS requires `Python 3.6`, `pip3` and `virtualenv` installed on the system.

1. `virtualenv .venv --python=python3`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

:warning: WARNING :warning:

Since TensorFlow 2.x is not compatible with TensorFlow 1.x, this implementation is tested only on Python 3.6.
If you have a newer version of Python, `pip` won't be able to find `tensorflow==1.x`. 


## Run the agent

1. Edit `params.yml` file. Comments in the file illustrate the individual fields.
2. Run `python drills.py train scl`

For help, `python drills.py -help`

## How It Works
<img src="doc/img/drills-architecture.png" width="70%" style="display: block;  margin: 0 auto;">

There are two major components in DRiLLS framework: 

* **Logic Synthesis** environment: a setup of the design space exploration problem as a reinforcement learning task. The logic synthesis environment is implemented as a session in [drills/scl_session.py](drills/scl_session.py) and [drills/fpga_session.py](drills/fpga_session.py).
* **Reinforcement Learning** environment: it employs an *Advantage Actor Critic agent (A2C)* to navigate the environment searching for the best optimization at a given state. It is implemented in [drills/model.py](drills/model.py) and uses [drills/features.py](drills/features.py) to extract AIG features.

DRiLLS agent exploring the design space of [Max](https://github.com/lsils/benchmarks/blob/master/arithmetic/max.v) design.

![](https://media.giphy.com/media/XbbW4WjeLuqneVbGEU/giphy.gif)

For more details on the inner-workings of the framework, see Section 4 in [the paper](https://github.com/scale-lab/DRiLLS/blob/drills-preprint/doc/preprint/DRiLLS_preprint_AH.pdf).

## Reporting Bugs
Please, use [ISSUE_TEMPLATE/bug_report.md](.github/ISSUE_TEMPLATE/bug_report.md) to create an issue and describe your bug.

## Contributing
Below is a list of suggested contributions you can make. Before you work on any, it is advised that you create an issue using the [ISSUE_TEMPLATE/contribution.md](.github/ISSUE_TEMPLATE/contribution.md) to tell us what you plan to work on. This ensures that your work can be merged to the `master` branch in a timely manner.

### Modernize Tensorflow Implementation

Google has recently released [Dopamine](https://github.com/google/dopamine) which sets up a framework for researching reinforcement learning algorithms. A new version of DRiLLS would adopt Dopamine to make it easier to implement the model and session classes. If you are new to Dopamine and want to try it on a real use case, it would be a great fit for DRiLLS and will add a great value to our repository.

### Better Integration
The current implementation interacts with the logic synthesis environment using files. This affects the run time of the agent training as it tries to extract features and statistics through files. A better integrations keeps a session of `yosys` and `abc` where the design is loaded once in the beginning and the feature extraction (and results extraction) are retrieved through this open session.

### Study An Enhanced Model
The goal is to enhance the model architecture used in [drills/model.py]. An enhancement should give better results (less area **AND** meets timing constraints):
* Deeper network architecure. 
* Changing gamma rate.
* Changing learning rate.
* Improve normalization.

## Citation
```
@INPROCEEDINGS{9045559,
    author={A. {Hosny} and S. {Hashemi} and M. {Shalan} and S. {Reda}},
     booktitle={2020 25th Asia and South Pacific Design Automation Conference (ASP-DAC)},
     title={DRiLLS: Deep Reinforcement Learning for Logic Synthesis},
     year={2020},
     volume={},
     number={},
     pages={581-586},}
```

## License
BSD 3-Clause License. See [LICENSE](LICENSE) file
