# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from .base_model import BaseModel
from .gp.gp import GP
from .rf.rf import RF
from .gp.gpy_wgp import GPyGP
from .gp.gpy_mlp import GPyMLPGP

model_dict = {
        'gp'  : GP,
        'rf'  : RF,
        'gpy' : GPyGP, 
        'gpy_mlp' : GPyMLPGP
        }

def get_model(model_name : str, *params, **conf) -> BaseModel:
    assert model_name in model_dict, "model name %s not in model_dict"
    model_class = model_dict[model_name]
    return model_class(*params, **conf)
