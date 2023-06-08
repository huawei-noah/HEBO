# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from torch import nn as nn
from torch.nn import functional as F
import torch
import math
from torch.autograd import Variable
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, arch_spec: list, f_act=None):
        """
        A standard multi-layer perceptron.
        :param d_in: number of input features.
        :param d_out: number of output features.
        :param arch_spec: list containing the number of units in each hidden layer. If arch_spec == [], this is a
                          linear model.
        :param f_act: nonlinear activation function (if arch_spec != [])
        """
        super(MLP, self).__init__()
        

        self.arch_spec = arch_spec
        self.f_act = f_act
        self.is_linear = (arch_spec == [])  # no hidden layers --> linear model

        if not self.is_linear:
            assert f_act is not None

        # define the network
        if self.is_linear:
            self.fc = nn.ModuleList([nn.Linear(in_features=d_in, out_features=d_out)])
        else:
            self.fc = nn.ModuleList([nn.Linear(in_features=d_in, out_features=arch_spec[0])])
            for i in range(1, len(arch_spec)):
                self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=arch_spec[i]))
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))

    def forward(self, X,vars=None):
        
        Y = X
        if self.is_linear:
            Y = self.fc[0](Y)
        else:
            for layer in self.fc[:-1]:
                Y = self.f_act(layer(Y))
            Y = self.fc[-1](Y)
        return Y

