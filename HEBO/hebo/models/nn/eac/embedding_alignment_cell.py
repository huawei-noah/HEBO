# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

"""
To deal with conditional input features and convert them into d-dimensional
feature space for each stage of the sequential input data.  For example, in a
stage, the input feature is (O_1, O_2, O_3, p_1, p_2, p_3).  When O_1 is
enabled, both O_2 and O_3 are disabled. At this time, the feature p_1, which
corresponds to O_1 is activated, but p_2 and p_3 are dis-activated.
"""

import torch
from torch import FloatTensor, LongTensor
from torch import nn
from copy import deepcopy

from hebo.design_space.design_space import DesignSpace
from hebo.models.layers import EmbTransform, OneHotTransform
from hebo.models.util import construct_hidden

class EmbeddingAlignmentCell(nn.Module):
    def __init__(self,
                 space: DesignSpace,
                 num_hiddens: int=32,
                 num_layers: int=1,
                 out_features: int=64,
                 enum_trans: str='onehot'):
        super(EmbeddingAlignmentCell, self).__init__()
        self.space          = space         # feature space, inlcuding name and range
        self.num_hiddens    = num_hiddens
        self.num_layers     = num_layers
        self.out_features   = out_features  # the dimensionality to be converted
        self.enum_trans     = enum_trans    # to deal with enumerated features

        self.eff_dim        = self.num_cont
        if self.num_enum > 0:               # when there is enumerated feature
            assert self.num_enum == len(self.num_uniqs)
            if self.enum_trans == 'onehot':
                self.enum_layer = OneHotTransform(self.num_uniqs)
            elif self.enum_trans == 'embedding':
                self.enum_layer = EmbTransform(self.num_uniqs)
            else:
                raise RuntimeError(f'Unknown enum processing type {enum_trans}, '
                                   f'can only be [embedding|onehot (default)]')
            self.eff_dim   += self.enum_layer.num_out

        self.hidden_layer   = construct_hidden(dim=self.eff_dim,
                                               num_layers=self.num_layers,
                                               num_hiddens=self.num_hiddens)
        self.output_layer   = nn.Linear(self.num_hiddens, self.out_features)

    @property
    def stage(self):
        names = [name.split('#')[-1] for name in self.space.para_names]
        assert len(set(names)) == 1, 'More than one stage name in a stage.'
        return list(set(names))[0]

    @property
    def num_uniqs(self):
        return [len(self.space.paras[item].categories)
                for item in self.space.enum_names]

    @property
    def num_cont(self):
        return self.space.num_numeric

    @property
    def num_enum(self):
        return self.space.num_categorical

    def xtrans(self, Xc: FloatTensor, Xe: LongTensor):
        """To preprocess the numeric and enumerated parts of input data separately.
        Only data that belong to current stage is valid.
        """
        num_data    = Xe.shape[0]
        Xc_         = Xc.clone()
        Xe_         = Xe.clone()

        X = self.space.inverse_transform(Xc, Xe)
        if self.space.numeric_names:
            for i, name in enumerate(self.space.numeric_names):
                depend      = name.split('#')[0].split('@')[-1]
                valid       = torch.FloatTensor((X[self.stage].values == depend).astype(float))
                Xc_[:, i]  *= valid
        else:
            Xc_ = torch.zeros(num_data, 0)

        Xe_trans    = self.enum_layer(Xe_)
        start       = 0
        for i, name in enumerate(self.space.enum_names):
            if name == self.stage:
                start  += self.enum_layer.num_out_list[i]
            else:
                depend  = name.split('#')[0].split('@')[-1]
                valid   = torch.FloatTensor((X[self.stage].values == depend).astype(float)).view(-1, 1)
                end     = start + self.enum_layer.num_out_list[i]
                Xe_trans[:, start:end] *= valid
                start   = end
        return torch.cat([Xc_, Xe_trans], dim=1)

    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        output = self.xtrans(Xc, Xe)
        output = self.hidden_layer(output)
        output = self.output_layer(output)
        return output


class EmbeddingAlignmentCells(nn.Module):
    def __init__(self,
                 stages: list,
                 space: DesignSpace,
                 out_features: int=64,
                 num_hiddens: int=64,
                 num_layers: int=1,
                 enum_trans: str='onehot',
                 share_weights: bool=False):
        """
        To deal with sequential data where each data comes as stage by stage.
        An EAC is used in each stage, and the EACs can share weights or not.
        """
        super(EmbeddingAlignmentCells, self).__init__()
        self.stages         = stages
        self.space          = space
        self.out_features   = out_features
        self.num_hiddens    = num_hiddens
        self.num_layers     = num_layers
        self.enum_trans     = enum_trans
        self.share_weights  = share_weights

        self.eac_layer      = nn.ModuleList()
        for _, space in self.subspaces.items():
            self.eac_layer.append(
                EmbeddingAlignmentCell(space=space,
                                       num_hiddens=self.num_hiddens,
                                       num_layers=self.num_layers,
                                       out_features=self.out_features,
                                       enum_trans=self.enum_trans))
            if self.share_weights:
                assert self._check_share_weights(), \
                    'The stages should be aligned when using weight-shared EACs!'
                break

    def _check_share_weights(self) -> bool:
        configs = []
        for stage in self.stages:
            config  = []
            for para_config in self.space.para_config:
                if stage == para_config['name'].split('#')[-1]:
                    _config = deepcopy(para_config)
                    _config['name'] = _config['name'].replace(stage, 'Stage')
                    config.append(_config)
            configs.append(sorted(config, key=lambda x: x['name']))
        return configs[1:] == configs[:-1]

    @property
    def subspaces(self) -> dict:
        spaces = {}
        for stage in self.stages:
            configs         = [config for config in self.space.para_config
                               if stage == config['name'].split('#')[-1]]
            spaces[stage]   = DesignSpace().parse(configs)
        return spaces

    def forward(self, Xc: FloatTensor, Xe: LongTensor):
        X       = self.space.inverse_transform(Xc, Xe)
        outputs = []
        for i, (stage, space) in enumerate(self.subspaces.items()):
            Xc_, Xe_  = space.transform(X)
            if self.share_weights:
                outputs.append(self.eac_layer[0](Xc_, Xe_))
            else:
                outputs.append(self.eac_layer[i](Xc_, Xe_))
        return torch.stack(outputs, dim=0)

