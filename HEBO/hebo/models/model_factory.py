# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from .base_model import BaseModel
from .gp.svgp import SVGP
from .gp.svidkl import SVIDKL
from .gp.gp import GP
from .gp.gpy_wgp import GPyGP
from .gp.gpy_mlp import GPyMLPGP
from .rf.rf import RF

from .nn.mcbn import MCBNEnsemble
from .nn.sgld import pSGLDEnsemble
from .nn.deep_ensemble import DeepEnsemble
from .nn.eac.masked_deep_ensemble import MaskedDeepEnsemble
from .nn.fe_deep_ensemble import FeDeepEnsemble
from .nn.gumbel_linear import GumbelDeepEnsemble

try:
    from .boosting.catboost import CatBoost
    has_catboost = True
except ImportError:
    has_catboost = False

model_dict = {
        'svidkl'  : SVIDKL,
        'svgp'  : SVGP,
        'gp'  : GP,
        'gpy' : GPyGP,
        'gpy_mlp' : GPyMLPGP, 
        'rf'  : RF,
        'deep_ensemble' : DeepEnsemble,
        'psgld' : pSGLDEnsemble,
        'mcbn' : MCBNEnsemble, 
        'masked_deep_ensemble' : MaskedDeepEnsemble,
        'fe_deep_ensemble': FeDeepEnsemble, 
        'gumbel': GumbelDeepEnsemble, 
        }
if has_catboost:
    model_dict['catboost'] = CatBoost

model_names = [k for k in model_dict.keys()]

def get_model_class(model_name : str):
    if model_name == 'multi_task':
        return MultiTaskModel
    else:
        assert model_name in model_dict, f"model name {model_name} not in {model_names}"
        model_class = model_dict[model_name]
        return model_class


def get_model(model_name : str, *params, **conf) -> BaseModel:
    model_class = get_model_class(model_name)
    return model_class(*params, **conf)

class MultiTaskModel(BaseModel):
    """
    Simple multi-task wrapper for models
    """
    support_multi_output = True
    def __init__(self, 
            num_cont, 
            num_enum, 
            num_out, 
            **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.model_name = self.conf.get('base_model_name', 'gp')
        self.model_conf = {k : v for k, v in self.conf.items() if k != 'model_name'}
        self.models     = [model_dict[self.model_name](num_cont, num_enum, 1, **self.model_conf) for _ in range(num_out)]
    

    def fit(self, Xc, Xe, y):
        for i in range(self.num_out):
            self.models[i].fit(Xc, Xe, y[:, [i]])


    def predict(self, Xc, Xe):
        py_cache  = []
        ps2_cache = []
        for m in self.models:
            py, ps2 = m.predict(Xc, Xe)
            py_cache.append(py)
            ps2_cache.append(ps2)
        return torch.cat(py_cache, dim = 1), torch.cat(ps2_cache, dim = 1)

    @property
    def noise(self):
        return torch.FloatTensor([m.noise for m in self.models]).reshape(self.num_out)
