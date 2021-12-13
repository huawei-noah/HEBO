# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import FloatTensor, LongTensor

from hebo.design_space import DesignSpace
from hebo.models.nn.deep_ensemble import DeepEnsemble, BaseNet
from hebo.models.base_model import BaseModel
from hebo.models.layers import OneHotTransform, EmbTransform

class MaskedBaseNet(BaseNet):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.space  = conf.get('space',  None)
        self.stages = conf.get('stages', None)

    def xtrans(self, Xc_ : FloatTensor, Xe_ : LongTensor) -> FloatTensor:
        if self.space is None or self.stages is None:
            return super().xtrans(Xc_, Xe_)

        num_data = Xe_.shape[0]
        Xc       = Xc_.clone()
        Xe       = Xe_.clone()
        Xe_trans = self.enum_layer(Xe)
        params   = self.space.inverse_transform(Xc, Xe)
        for i, stage in enumerate(self.stages):
            stage_null  = params[stage] == 'null'
            rest_stages = self.stages[i+1:]
            params.loc[stage_null, rest_stages] = 'null'

        if self.space.numeric_names:
            for i, name in enumerate(self.space.numeric_names):
                stage     = name.split('#')[-1]
                depend    = name.split('#')[0].split('@')[-1]
                valid     = torch.FloatTensor((params[stage].values == depend).astype(float))
                Xc[:, i] *= valid
        else:
            Xc = torch.zeros(num_data, 0)
        
        start_idx = 0
        for i, name in enumerate(self.space.enum_names):
            if name in self.stages:
                start_idx += self.enum_layer.num_out_list[i]
            else:
                stage   = name.split('#')[-1]
                depend  = name.split('#')[0].split('@')[-1]
                valid   = torch.FloatTensor((params[stage].values == depend).astype(float)).view(-1, 1)
                end_idx = start_idx + self.enum_layer.num_out_list[i]
                Xe_trans[:, start_idx:end_idx] *= valid
                start_idx = end_idx
        return torch.cat([Xc, Xe_trans], axis = 1)

class MaskedDeepEnsemble(DeepEnsemble):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.stages      = self.conf.get('stages', None)
        self.space       = self.conf.get('space', None)
        self.basenet_cls = MaskedBaseNet
