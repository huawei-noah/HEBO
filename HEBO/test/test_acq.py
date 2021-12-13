# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import torch
import pytest

from hebo.models.rf.rf import RF
from hebo.acquisitions.acq import Mean, Sigma, LCB, MOMeanSigmaLCB, MACE, GeneralAcq, SingleObjectiveAcq, NoisyAcq

X = torch.randn(10, 1)
y = X
model = RF(1, 0, 1)
model.fit(X, None, y)

@pytest.mark.parametrize('acq_cls', 
        [Mean, Sigma, LCB])
def test_acq(acq_cls):
    acq   = acq_cls(model, best_y = 0.)
    acq_v = acq(X, None)
    if isinstance(acq, SingleObjectiveAcq):
        assert acq.num_obj    == 1
        assert acq.num_constr == 0
    assert acq_v.shape[1] == acq.num_obj + acq.num_constr

def test_mo_acq():
    acq   = MOMeanSigmaLCB(model, best_y = 0.)
    acq_v = acq(X, None)
    assert torch.isfinite(acq_v).all()
    assert acq.num_obj    == 2
    assert acq.num_constr == 1

def test_mace():
    acq   = MACE(model, best_y = 0.)
    acq_v = acq(X, None)
    assert torch.isfinite(acq_v).all()
    assert acq.num_obj    == 3
    assert acq.num_constr == 0
    
def test_general():
    acq   = GeneralAcq(model, 1, 0)
    acq_v = acq(X, None)
    assert torch.isfinite(acq_v).all()
    assert acq.num_obj    == 1
    assert acq.num_constr == 0

def test_noisy():
    acq   = NoisyAcq(model, 1, 0)
    acq_v = acq(X, None)
    assert torch.isfinite(acq_v).all()
    assert acq.num_obj    == 1
    assert acq.num_constr == 0
