# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import torch
from torch import nn
import numpy as np
from typing import Optional, Tuple, Union
from .modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    CausalLMOutputWithPastMasked,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from .hooks import (
    AttentionHooks,
    MLPHooks,
    NormalizationHooks,
    TransformerHooks,
)
from .modeling_gpt_neox import GPTNeoXForCausalLM
from .modeling_gpt_bigcode import GPTBigCodeForCausalLM
# from .modeling_gpt2 import GPT2LMHeadModel


def apply_masked_z_norm(input, mask):
    # input: (b,seq,1)
    input = input.squeeze()
    _mean = ((input * mask).sum(axis=-1) / mask.sum(axis=-1)).view(-1,1)
    _std = []
    bs = input.size(0)
    for i in range(bs):
        _std.append(input[i,mask[i]].std())
    _std = torch.Tensor(_std).type(torch.float16).to(input.device).view(-1,1) + 1e-12
    
    # z normalize
    input = (input - _mean) / _std
    # rescale to 0,1
    _min = [input[i,mask[i]].min() for i in range(bs)]
    _max = [input[i,mask[i]].max() for i in range(bs)]
    _min = torch.Tensor(_min).type(torch.float16).to(input.device).view(-1,1)
    _max = torch.Tensor(_max).type(torch.float16).to(input.device).view(-1,1)

    rescaled = (input - _min) / (_max - _min)
    rescaled = (rescaled * mask).view(bs,-1,1)
    return rescaled

def apply_masked_rescaling(input, mask):
    # input: (b,seq,1)
    input = input.squeeze()
    bs = input.size(0)
    # rescale to 0,1
    _min = [input[i,mask[i]].min() for i in range(bs)]
    _max = [input[i,mask[i]].max() for i in range(bs)]
    _min = torch.Tensor(_min).type(torch.float16).to(input.device).view(-1,1)
    _max = torch.Tensor(_max).type(torch.float16).to(input.device).view(-1,1)

    rescaled = (input - _min) / (_max - _min)
    rescaled = (rescaled * mask).view(bs,-1,1)
    return rescaled

class SimpleMaskLayer(torch.nn.Module):
    def __init__(self, config, out_drpt=True, clamp=False, drpt=0.1, activation='relu'):
        super().__init__()
        self.w = torch.nn.Linear(config.hidden_size, 1)
        self.activation_name = activation
        if activation == 'relu':
            self.actv = torch.nn.ReLU()
        elif activation == 'znorm':
            self.actv = apply_masked_z_norm
        elif activation == 'resc':
            self.actv = apply_masked_rescaling
        elif activation == 'none':
            self.actv = None
        self.do_out_drpt = out_drpt
        self.do_clamp = clamp
        if self.do_out_drpt:
            self.out_drop = torch.nn.Dropout(p=drpt)
        
    def forward(self, input, mask=None):
        h = self.w(input)
        if   self.activation_name == 'relu':
            h = self.actv(h)
        elif self.activation_name == 'znorm' or self.activation_name == 'resc':
            h = self.actv(h, mask)
        if self.do_clamp:
            h = torch.clamp(h, max=1.0)
        if self.do_out_drpt:
            h = self.out_drop(h)
        return h

class SimpleMaskAllLayers(torch.nn.Module):
    def __init__(self, config, drpt=0.1, mixer_activation="relu", layer_activation="relu"):
        super().__init__()
        self.w_per_layer = torch.nn.ModuleList(
            [SimpleMaskLayer(config, out_drpt=True, clamp=False, drpt=drpt, activation=layer_activation) \
             for _ in range(config.num_hidden_layers)])
        self.mixer = torch.nn.Linear(config.num_hidden_layers, 1, bias=False)
        self.activation_name = mixer_activation
        if mixer_activation == 'relu':
            self.actv = torch.nn.ReLU()
        elif mixer_activation == 'znorm':
            self.actv = apply_masked_z_norm
        elif mixer_activation == 'resc':
            self.actv = apply_masked_rescaling
        elif mixer_activation == 'none':
            self.actv = None
        ln_eps = 0
        if config.model_type in ["gpt_bigcode","gpt2","gptj"]:
            ln_eps = config.layer_norm_epsilon
        else:
            ln_eps = config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=ln_eps)
        self.mix_drop = torch.nn.Dropout(p=drpt)

    def forward(self, input_per_layer, mask=None):
        h_per_layer = []
        for i, inp in enumerate(input_per_layer):
            inp = self.layer_norm(inp)
            h_per_layer.append(self.w_per_layer[i](inp, mask))
        #
        h = torch.concat(h_per_layer, dim=-1)  # (b,seq,L)
        h = self.mixer(h)  # (b,seq,1)  # Normalize?
        if   self.activation_name == 'relu':
            h = self.actv(h)
        elif self.activation_name == 'znorm' or self.activation_name == 'resc':
            h = self.actv(h, mask)
        h = self.mix_drop(h)
        return h



class DoubleFFMaskLayer(torch.nn.Module):
    def __init__(self, config, out_drpt=True, clamp=False):
        super().__init__()
        self.ff1 = torch.nn.Linear(config.hidden_size, 200)  # how is 200 decided?
        self.ff2 = torch.nn.Linear(200, 1)
        self.relu = torch.nn.ReLU()
        self.ff_drop = torch.nn.Dropout(p=0.1)
        self.do_out_drpt = out_drpt
        self.do_clamp = clamp
        if self.do_out_drpt:
            self.out_drop = torch.nn.Dropout(p=0.1)
        
    def forward(self, input):
        h = self.ff1(input)
        h = self.ff_drop(h)
        h = self.ff2(h)
        h = self.relu(h)
        # h = torch.sigmoid(h)
        if self.do_out_drpt:
            h = self.out_drop(h)
        # if self.do_clamp:
        #     h = torch.clamp(h, max=1.0)
        return h


class DoubleFFMaskAllLayers(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_per_layer = torch.nn.ModuleList([DoubleFFMaskLayer(config,True,False) for _ in range(config.num_hidden_layers)])
        self.mixer = torch.nn.Linear(config.num_hidden_layers, 1, bias=False)
        self.relu = torch.nn.ReLU()
        ln_eps = 0
        if config.model_type == "gpt_bigcode" or config.model_type == "gpt2":
            ln_eps = config.layer_norm_epsilon
        else:
            ln_eps = config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=ln_eps)
        # self.mix_drop = torch.nn.Dropout(p=0.1)
        
    def forward(self, input_per_layer):
        h_per_layer = []
        for i,inp in enumerate(input_per_layer):
            inp = self.layer_norm(inp)
            h_per_layer.append(self.w_per_layer[i](inp))
            
        h = torch.concat(h_per_layer,dim=-1) # (b,seq,L)
        # h = self.mix_drop(h)
        h = self.mixer(h) # (b,seq,1)
        h = self.relu(h)
        # h = torch.clamp(h, max=1.0)
        return h

## ablation masks

class RandomMask:
    def __init__(self, config):
        np.random.seed(42)
        self.rng_gen = np.random.default_rng()

    def get_mask(self, input, **kwargs): # logits form last layer
        bs,seqlen = input.shape[:2]
        mask = self.rng_gen.uniform(0.0,1.0,size=[bs,seqlen])
        return torch.tensor(mask).to(input.device)
    
class BinaryMaskLast(torch.nn.Module):
    def __init__(self, config, out_drpt=True):
        super().__init__()
        self.w = torch.nn.Linear(config.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.do_out_drpt = out_drpt
        if self.do_out_drpt:
            self.out_drop = torch.nn.Dropout(p=0.1)
        
    def forward(self, input):
        h = self.w(input)
        h = self.relu(h)
        if self.do_out_drpt:
            h = self.out_drop(h)
        h = torch.sign(h)
        return h

class BinaryMaskAllLayers(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_per_layer = torch.nn.ModuleList(
            [SimpleMaskLayer(config, True, False) for _ in range(config.num_hidden_layers)])
        self.mixer = torch.nn.Linear(config.num_hidden_layers, 1, bias=False)
        self.relu = torch.nn.ReLU()
        ln_eps = 0
        if config.model_type == "gpt_bigcode" or config.model_type == "gpt2":
            ln_eps = config.layer_norm_epsilon
        else:
            ln_eps = config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=ln_eps)
        self.mix_drop = torch.nn.Dropout(p=0.1)

    def forward(self, input_per_layer):
        h_per_layer = []
        for i, inp in enumerate(input_per_layer):
            inp = self.layer_norm(inp)
            h_per_layer.append(self.w_per_layer[i](inp))

        h = torch.concat(h_per_layer, dim=-1)  # (b,seq,L)
        h = self.mix_drop(h)
        h = self.mixer(h)  # (b,seq,1)  # Normalize?
        h = self.relu(h)
        h = torch.sign(h)
        return h
