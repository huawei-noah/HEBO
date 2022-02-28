# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

"""
A package to manage DRiLLS implementation; utilizing ABC and Tensorflow
...

Classes:
--------
SCLSession: to manage the logic synthesis environment when using a standard cell library
FPGASession: to manage the logic synthesis environment when using FPGAs
A2C: contains the deep neural network model (Advantage Actor Critic)

Helpers:
--------
yosys_stats: extract design metrics using yosys
abc_stats: extract design metrics using ABC
extract_features: extract design features used as input to the model
"""