# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from .base_model import BaseModel
from .gp.gp import GP
from .gp.gpy_wgp import GPyGP
from .gp.gpy_mlp import GPyMLPGP
from .rf.rf import RF
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
        'gp'  : GP,
        'gpy' : GPyGP,
        'gpy_mlp' : GPyMLPGP, 
        'rf'  : RF,
        'deep_ensemble' : DeepEnsemble,
        'masked_deep_ensemble' : MaskedDeepEnsemble,
        'fe_deep_ensemble': FeDeepEnsemble, 
        'gumbel': GumbelDeepEnsemble, 
        }
if has_catboost:
    model_dict['catboost'] = CatBoost


def get_model(model_name : str, *params, **conf) -> BaseModel:
    assert model_name in model_dict, "model name %s not in model_dict"
    model_class = model_dict[model_name]
    return model_class(*params, **conf)

def get_model_class(model_name : str, *params, **conf):
    assert model_name in model_dict, "model name %s not in model_dict"
    model_class = model_dict[model_name]
    return model_class
