# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import torch
import torch.nn as nn
import pytest

from hebo.models.layers import OneHotTransform, EmbTransform

def test_encoders():
    layer1 = EmbTransform([5, 5], emb_sizes = [1, 1])
    layer2 = EmbTransform([5, 5], emb_sizes = [2, 2])
    layer3 = OneHotTransform([5, 5])
    layer4 = EmbTransform([5, 5]) 

    xe     = torch.randint(5, (10, 2))
    assert(layer1(xe).shape[1] == 2)
    assert(layer1.num_out == 2)

    assert(layer2(xe).shape[1] == 4)
    assert(layer2.num_out == 4)

    assert(layer4(xe).shape[1] == layer4.num_out)

    assert(layer3(xe).shape[1] == 10)
    assert(layer3.num_out == 10)

    assert(torch.all((layer3(xe) == 0) | (layer3(xe) == 1)))
    assert(torch.all(layer3(xe).sum(dim = 1) == xe.shape[1]))

    model1 = nn.Sequential(
            OneHotTransform([5, 5]), 
            nn.Linear(10, 1))
    model2 = nn.Sequential(
            EmbTransform([5, 5], emb_sizes = [2, 2]), 
            nn.Linear(4, 1))
    assert(model1(xe).shape == model2(xe).shape)

    with pytest.raises((RuntimeError, IndexError)):
        xe = torch.randint(50, (100, 2))
        layer1(xe)

    with pytest.raises(RuntimeError):
        xe = torch.randint(50, (100, 2))
        layer3(xe)
