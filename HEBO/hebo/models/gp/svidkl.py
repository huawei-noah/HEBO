# TODO:
# spectral-normalization
import torch.nn as nn
from torch.nn.utils import spectral_norm

from copy import deepcopy
from gpytorch.kernels import ScaleKernel, MaternKernel

from .gp_util import DummyFeatureExtractor
from .svgp import SVGP

def construct_hidden(dim, num_layers, num_hiddens, act = nn.ReLU(), sn_norm = None) -> nn.Module:
    layers = [SNLinear(dim, num_hiddens, sn_norm), act]
    for i in range(num_layers - 1):
        layers.append(SNLinear(num_hiddens, num_hiddens, sn_norm))
        layers.append(act)
    return nn.Sequential(*layers)

class SNLinear(nn.Module):
    def __init__(self, in_features, out_features, sn_norm = None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sn_norm      = sn_norm
        if self.sn_norm:
            self.linear = spectral_norm(nn.Linear(in_features, out_features))
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        proj = self.linear(x)
        if self.sn_norm:
            proj *= self.sn_norm
        return proj

    def __repr__(self):
        return f'SNLinear(in_features = {self.in_features}, out_features = {self.out_features}, sn_norm = {self.sn_norm})'

class DKLFe(nn.Module):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__()
        self.num_hiddens = conf.get('num_hiddens', 64)
        self.num_layers  = conf.get('num_layers', 2) 
        self.act         = conf.get('act', nn.LeakyReLU())
        self.sn_norm     = conf.get('sn_norm') # no spectral normalization
        self.emb_trans   = DummyFeatureExtractor(num_cont, num_enum, conf.get('num_uniqs'), conf.get('emb_sizes'))

        self.fe          = construct_hidden(self.emb_trans.total_dim, self.num_layers, self.num_hiddens, self.act, self.sn_norm)
        self.total_dim   = self.num_hiddens
    
    def forward(self, x, xe):
        x_all = self.emb_trans(x, xe)
        return self.fe(x_all)

class SVIDKL(SVGP):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        new_conf = deepcopy(conf)
        new_conf.setdefault('ard_kernel', False) # no need to use ARD kernel when there's a NN feature extractor
        new_conf.setdefault('fe', DKLFe(num_cont, num_enum, num_out, **new_conf))
        new_conf.setdefault('kern', ScaleKernel(MaternKernel(nu = 2.5)))
        super().__init__(num_cont, num_enum, num_out, **new_conf)
