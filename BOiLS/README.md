# BOiLS: Bayesian Optimisation for Logic Synthesis

Logic synthesis oriented Bayesian optimsation library developped by _Huawei Noah's Ark lab_. Developped to carry out 
the experiments reported in [BOiLS: Bayesian Optimisation for Logic Synthesis](https://ieeexplore.ieee.org/document/9774632), 
accepted at DATE22 conference.

<p align="center">
    <img src="./results/sample-eff-1.png" alt="drawing" width="500"/>
</p>

## Contributors

**Antoine Grosnit**, **Cedric Malherbe**, **Rasul Tutunov**, **Xingchen Wan**, **Jun Wang**, **Haitham Bou-Ammar**
 -- _Huawei Noah's Ark lab_.

## Setup
Our experiments were performed on two machines with **Intel Xeon CPU E5-2699 v4@2.20GHz**, 64GB RAM, running
**Ubuntu 18.04.4 LTS** and equipped with one **NVIDIA Tesla
V100** GPU. All algorithms were implemented in **Python 3.7** relying on `ABC v1.01`.

### Environment
- Install yosys
```shell script
sudo apt-get update -y
sudo apt-get install -y yosys
```

- Create Python 3.7 venv

```shell script
# Create virtualenv
python3.7 -m venv ./venv

# Activate venv
source venv/bin/activate

# Try installing requirements
pip install ./requirements.txt  # if getting issues with torch installation visit: https://pytorch.org/get-started/previous-versions/

#----- Begin Graph-RL: if you need to run Graph-RL experiments, you need to install the following (you can skip this if BOiLS is only what you need): 
# follow instructions from: https://github.com/krzhu/abc_py
#-----  End Graph-RL -----
```

### Dataset
Dataset and results should be **stored in the same directory** `STORAGE_DIRECTORY`: run `python utils_save.py` and follow 
the instructions given in the `FileNotFoundError` message. Rerun `python utils_save.py` to check 
where the data should be saved (`DATA_PATH`) and where the results will be stored.

- download the circuits from [EPFL Combinatorial Benchmark Suite](https://github.com/lsils/benchmarks) and put them 
(only the "*.blif" are needed) in 
`DATA_PATH/benchmark_blif/` (not in a subfolder as the code will look for the circuits directly as 
`DATA_PATH/benchmark_blif/*.blif`).

### Setup sanity-check for fair comparison
If comparing with our reported results, run the following in your environment and make sure the output statistics are the same:
```shell script
DATA_PATH=... # change with your DATA_PATH
yosys-abc  -c "read $DATA_PATH/benchmark_blif/sqrt.blif; strash; balance; rewrite -z; if -K 6; print_stats;"
# Should output:
#  top                           : i/o =  128/   64  lat =    0  nd =  4005  edge =  19803  aig  = 29793  lev = 1023
```

---
## Run experiments

The code is organised in a modular way, providing coherent API for all optimisation methods. Third-party libraries used for the baseline implementations can be found in 
the [resources](./resources) directory, while the scripts to run the synthesis flow optimisation experiments are in the 
[core](./core) folder. The only exception to this organisation is for `DRiLLS` algorithm whise implementation is stored in [DRiLLS](./DRiLLS).

#### Run BOiLS
**BOiLS** can be run as shown below to find a sequence of logic synthesis primitives optimising the area / delay of a given circuit (e.g. `log2.blif` from EPFL benchmark). 

```shell script
python ./core/algos/bo/boils/main_multi_boils.py --designs_group_id log2 --n_parallel $n_parallel 1 \
                      --seq_length 20 --mapping fpga --action_space_id extended --ref_abc_seq resyn2 \
                      --n_total_evals 200 --n_initial 20 --device 0 --lut_inputs 4 --use_yosys 1  \
                      --standardise --ard --acq ei --kernel_type ssk \
                      --length_init_discrete_factor .666 --failtol 40 \
                      --objective area \
                      --seed 0"
```
Meaning of all the parameters are provided in the script: [./core/algos/bo/hebo/multi_hebo_exp.sh](core/algos/bo/hebo/multi_hebo_exp.sh). We created similar scripts for a wide set of optimisers, as detailed in the following section.


#### Setup to run COMBO
To run sequence optimisation using [**COMBO**](https://github.com/QUVA-Lab/COMBO) you need to download code of the 
official implementation, and to put it in the `./resources/` folder. 

```shell script
cd resources
wget https://github.com/QUVA-Lab/COMBO/archive/refs/heads/master.zip
unzip master.zip
mv COMBO-master/ COMBO
```

#### Optimisation strategies

| Algorithm                      | Implementation                                     | Optimisation script                                                                      | Comment                                                                                                                                                                                                                                              |
|--------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Reinforcement Learning**     |                                                    |                                                                                          |                                                                                                                                                                                                                                                      |
| DRiLLS                         | [./DRiLLS](./DRiLLS)                               | [./DRiLLS/drills_script.sh](./DRiLLS/drills_script.sh)                                   | Implementation was taken from the [DRiLLS official repository](https://github.com/scale-lab/DRiLLS) . The code is adapted to run with PPO and A2C update rules, using [stable-baselines](https://github.com/hill-a/stable-baselines)  library.       |
| Graph-RL                       | [./resources/abcRL](./resources/abcRL)             | [./core/algos/GRiLLS/multi_grills_exp.sh](core/algos/GRiLLS/multi_grills_exp.sh)         | Implementation was taken from the [abcRL official repository](https://github.com/krzhu/abcRL). The reward function has been changed so that agents optimise the QoR improvement on both **area** and **delay**.                                      |
| **Bayesian optimisation**      |                                                    |                                                                                          |                                                                                                                                                                                                                                                      |
| Standard BO                    | [./core/algos/bo/hebo](core/algos/bo/hebo)         | [./core/algos/bo/hebo/multi_hebo_exp.sh](core/algos/bo/hebo/multi_hebo_exp.sh)           | Implementation was taken from [HEBO](https://github.com/huawei-noah/noah-research/tree/master/BO/HEBO).                                                                                                                                              |
| COMBO                          | [./core/algos/bo/combo](core/algos/bo/combo)       | [./core/algos/bo/boils/multi_combo_exp.sh](core/algos/bo/boils/multi_combo_exp.sh)       | Using official COMBO implementation: [COMBO](https://github.com/QUVA-Lab/COMBO).                                                                                                                                                                     |
| BOiLS                          | [./core/algos/bo/boils](core/algos/bo/boils)       | [./core/algos/bo/boils/multiseq_boils_exp.sh](core/algos/bo/boils/multiseq_boils_exp.sh) | Adaptation of [Casmopolitan](https://github.com/xingchenwan/Casmopolitan) implementation using a string-subsequence kernel (SSK) in the surrogate model. The SSK is a pytorch rewriting of [BOSS](https://github.com/henrymoss/BOSS) implementation. |
| **Genetic Algorithm**          |                                                    |                                                                                          |                                                                                                                                                                                                                                                      |
| Simple Genetic Algorithm       | [./core/algos/genetic/sga](core/algos/genetic/sga) | [./core/algos/genetic/sga/multi_sga_exp.sh](core/algos/genetic/sga/multi_sga_exp.sh)     | Used simple genetic algorithm from [geneticalgorithm2](https://pypi.org/project/geneticalgorithm2/).                                                                                                                                                 |
| **Random Search**              |                                                    |                                                                                          |                                                                                                                                                                                                                                                      |
| Latin Hypercube Sampling (LHS) | [./core/algos/random](core/algos/random)           | [./core/algos/random/multi_random_exp.sh](core/algos/random/multi_random_exp.sh)         | Used LHS from [pymoo]().                                                                                                                                                                                                                             |
| **Greedy Search**              |                                                    |                                                                                          |                                                                                                                                                                                                                                                      |
| Greedy oprimisation            | [./core/algos/greedy](core/algos/greedy)           | [./core/algos/greedy/main_greedy_exp.sh](core/algos/greedy/main_greedy_exp.sh)           | Implementation from scratch ([code](core/algos/greedy/greedy_exp.py)).                                                                                                                                                                               |

## Cite Us

**Grosnit, Antoine, et al. "Bayesian Optimisation for Logic Synthesis", DATE (2022).**

#### BibTex
```
@inproceedings{DBLP:conf/date/GrosnitMTWWB22,
  author    = {Antoine Grosnit and
               C{\'{e}}dric Malherbe and
               Rasul Tutunov and
               Xingchen Wan and
               Jun Wang and
               Haitham Bou{-}Ammar},
  editor    = {Cristiana Bolchini and
               Ingrid Verbauwhede and
               Ioana Vatajelu},
  title     = {BOiLS: Bayesian Optimisation for Logic Synthesis},
  booktitle = {2022 Design, Automation {\&} Test in Europe Conference {\&}
               Exhibition, {DATE} 2022, Antwerp, Belgium, March 14-23, 2022},
  pages     = {1193--1196},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.23919/DATE54114.2022.9774632},
  doi       = {10.23919/DATE54114.2022.9774632},
  timestamp = {Wed, 25 May 2022 22:56:20 +0200},
  biburl    = {https://dblp.org/rec/conf/date/GrosnitMTWWB22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement

- **Stable baselines**: A. Hill, A. Raffin _et al._, ''Stable Baselines,'' https://github.com/hill-a/stable-baselines, 2018.

- **DRiLLS**: H. Abdelrahman, S. Hashemi _et al._ ''DRiLLS: Deep reinforcement learning for logic synthesis,'' 2020
25th Asia and South Pacific Design Automation Conference (ASP-DAC)

- **abcRL**: K. Zhu _et al._, ''Exploring Logic Optimizations with Reinforcement Learning and Graph Convolutional
Network,'' Proceedings of the 2020 ACM/IEEE Workshop on Machine Learning for CAD, 2020.
 
- **HEBO**: A. Cowen-Rivers _et al._, ''An Empirical Study of Assumptions in Bayesian Optimisation,''
arXiv preprint arXiv:2012.03826, 2020
 
- **Casmopolitan**: X. Wan _et al._, ''Think Global and Act Local: Bayesian Optimisation over High-Dimensional 
Categorical and Mixed Search Spaces,'' International Conference on Machine Learning (ICML), 2021.

- **BOSS**: H. B. Moss, ''BOSS: Bayesian Optimization over String Spaces'', NeurIPS, 2020.

- **geneticalgorithm2**: D. Pascal, ''geneticalgorithm2 (v.6.2.12)'', 
https://github.com/PasaOpasen/geneticalgorithm2, 2021.

- **pymoo**: J. Blank and K. Deb, ''pymoo: Multi-Objective Optimization in Python,'' IEEE Access, 2020.

