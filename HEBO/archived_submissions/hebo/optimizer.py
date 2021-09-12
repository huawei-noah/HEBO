# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy  as np
import pandas as pd
import torch
from bo.design_space.design_space import DesignSpace
from bo.models.gp.gp import GP
from bo.acquisitions.acq import LCB, Mean, Sigma, MOMeanSigmaLCB
from bo.optimizers.evolution_optimizer import EvolutionOpt

torch.set_num_threads(min(1, torch.get_num_threads()))

from optimizer_mace import MACEBO

if __name__ == "__main__":
    experiment_main(MACEBO)
