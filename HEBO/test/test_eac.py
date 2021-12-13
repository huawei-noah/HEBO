# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os, platform
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest
import pandas as pd
import numpy as np

import torch
from torch import FloatTensor
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hebo.models.nn.eac import MaskedDeepEnsemble, EACEnsemble
from hebo.design_space.design_space import DesignSpace

@pytest.mark.parametrize('output_noise', [True, False], ids = ['nll', 'mse'])
@pytest.mark.parametrize('model_type', ['rnn', 'lstm', 'transformer', 'mlp'])
@pytest.mark.parametrize('model_cls', [MaskedDeepEnsemble, EACEnsemble])
@pytest.mark.parametrize('num_processes', [1, 5])
@pytest.mark.parametrize('share_weights', [False, True])
def test_eacrnn(output_noise, model_type, model_cls, num_processes, share_weights):
    if platform.system() == 'Linux' and num_processes > 1:
        pytest.skip('Skip multiprocessing test in Linux')
    space = DesignSpace().parse([
        {'name': 'Stage0', 'type': 'cat', 'categories': ['RRELU', 'LeakyRELU']},
        {'name': 'Stage1', 'type': 'cat', 'categories': ['RRELU', 'LeakyRELU']},
        {'name': 'lower@RRELU#Stage0', 'type': 'num', 'lb': 0.01, 'ub': 0.125},
        {'name': 'upper@RRELU#Stage0', 'type': 'num', 'lb': 0.15, 'ub': 0.5},
        {'name': 'dummy@RRELU#Stage0', 'type': 'cat', 'categories' : ['a', 'b', 'c', 'd']},
        {'name': 'dummy@RRELU#Stage1', 'type': 'cat', 'categories' : ['a', 'b', 'c', 'd']},
        {'name': 'lower@RRELU#Stage1', 'type': 'num', 'lb': 0.01, 'ub': 0.125},
        {'name': 'upper@RRELU#Stage1', 'type': 'num', 'lb': 0.15, 'ub': 0.5},
        {'name': 'negative_slope@LeakyRELU#Stage0', 'type': 'num', 'lb': 0.01, 'ub': 0.125},
        {'name': 'negative_slope@LeakyRELU#Stage1', 'type': 'num', 'lb': 0.01, 'ub': 0.125}
    ])
    num_uniqs = [len(space.paras[item].categories) for item in space.enum_names]

    df_X    = space.sample(20)
    df_X['dummy@RRELU#Stage0'] = 'a' # XXX: other categories not seen in training dat
    df_X['dummy@RRELU#Stage1'] = 'a' # XXX: other categories not seen in training dat
    y       = torch.from_numpy(train_boston_housing(df_X))
    Xc, Xe  = space.transform(df_X)

    model = model_cls(Xc.shape[1], Xe.shape[1], 1, 
            num_uniqs     = num_uniqs,
            space         = space,
            stages        = ['Stage0', 'Stage1'],
            model_type    = model_type,
            output_noise  = output_noise,
            num_processes = num_processes,
            share_weights = share_weights,
            num_epochs    = 1)
    model.fit(Xc, Xe, y)
    py, _ = model.predict(Xc, Xe)
    assert np.isfinite(r2_score(y, py.detach().numpy()))

    para_tst           = space.sample(20)
    para_tst['Stage0'] = 'LeakyRELU'
    para_tst['Stage1'] = 'LeakyRELU'
    para_tst[['negative_slope@LeakyRELU#Stage0', 'negative_slope@LeakyRELU#Stage1']] = 1e-2
    X, Xe = space.transform(para_tst)
    with torch.no_grad():
        py, ps2 = model.predict(X, Xe)
        assert py.var()  / y.var() < 1e-4
        assert ps2.var() / y.var() < 1e-4

@pytest.mark.parametrize('model_cls', [MaskedDeepEnsemble, EACEnsemble])
def test_null(model_cls):
    space = DesignSpace().parse([
        {'name' : 'S0', 'type' : 'cat' ,  'categories' : ['a', 'b']}, 
        {'name' : 'S1', 'type' : 'cat' ,  'categories' : ['a', 'b', 'null']}, 
        {'name' : 'S2', 'type' : 'cat' ,  'categories' : ['a', 'b', 'null']}, 
        {'name' : 'S3', 'type' : 'cat' ,  'categories' : ['a', 'b', 'null']}, 
        {'name' : 'v@a#S0', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
        {'name' : 'v@a#S1', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
        {'name' : 'v@a#S2', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
        {'name' : 'v@a#S3', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
        ])
    num_uniqs    = [len(space.paras[item].categories) for item in space.enum_names]
    num_data     = 50
    params       = space.sample(num_data)
    params['S0'] = 'b'    # XXX: v@a#S0 is invalidated
    params['S1'] = 'b'    # XXX: v@a#S1 is invlidated
    params['S2'] = 'null' # XXX: V@a#S2 is invalidated and S3
    params['S3'] = 'a'    # XXX: S3 ins invalidated, so that V@a#S3 is invalidate
    X, Xe = space.transform(params)
    y     = 100 * torch.randn(num_data, 1)
    model = model_cls(X.shape[1], Xe.shape[1], 1, 
            num_uniqs  = num_uniqs, 
            space      = space,
            stages     = ['S0', 'S1', 'S2', 'S3'],
            num_epochs = 0)
    model.fit(X, Xe, y)

    with torch.no_grad():
        py, ps2 = model.predict(X, Xe)
        assert py.var()  < 1e-3
        assert ps2.var() < 1e-3

@pytest.mark.parametrize('model_cls', [MaskedDeepEnsemble, EACEnsemble])
def test_thompson_sampling(model_cls):
    space = DesignSpace().parse([
        {'name' : 'S0', 'type' : 'cat' ,  'categories' : ['a', 'b']}, 
        {'name' : 'v@a#S0', 'type' : 'num', 'lb' : 0, 'ub' : 1}, 
        ])
    num_uniqs    = [len(space.paras[item].categories) for item in space.enum_names]
    params       = space.sample(30)
    X, Xe = space.transform(params)
    y     = torch.randn(30, 1)
    model = model_cls(X.shape[1], Xe.shape[1], 1, num_uniqs  = num_uniqs, space = space, stages = ['S0'], num_epochs = 1)
    assert model.support_ts
    model.fit(X, Xe, y)

    with torch.no_grad():
        fs      = [model.sample_f() for _ in range(model.num_ensembles)]
        samp    = torch.stack([f(X, Xe) for f in fs], axis = 0).mean(axis = 0)
        pred, _ = model.predict(X, Xe)
        r2      = r2_score(pred.numpy(), samp.numpy())
        assert pred.shape == samp.shape
        assert torch.isfinite(samp).all()
        assert r2 > 0.5

@pytest.mark.parametrize('model_cls', [MaskedDeepEnsemble, EACEnsemble])
def test_enum_only(model_cls):
    space = DesignSpace().parse([
        {'name' : 'S0', 'type' : 'cat' ,  'categories' : ['a', 'b']}, 
        {'name' : 'v@a#S0', 'type' : 'cat', 'categories' : ['c', 'd']}, 
        ])
    num_uniqs    = [len(space.paras[item].categories) for item in space.enum_names]
    params       = space.sample(30)
    X, Xe = space.transform(params)
    y     = torch.randn(30, 1)
    model = model_cls(X.shape[1], Xe.shape[1], 1, num_uniqs  = num_uniqs, space = space, stages = ['S0'], num_epochs = 1)
    assert model.support_ts
    model.fit(X, Xe, y)

    with torch.no_grad():
        py, ps2 = model.predict(X, Xe)
        assert torch.isfinite(py).all()
        assert (ps2 > 0).all()

@pytest.mark.parametrize('enum_trans', ['embedding', 'onehot', 'unspported'])
def test_enum_layer(enum_trans):
    space = DesignSpace().parse([
        {'name' : 'S0', 'type' : 'cat' ,  'categories' : ['a', 'b']}, 
        {'name' : 'v@a#S0', 'type' : 'num', 'lb' : -1, 'ub' : 1}, 
        {'name' : 'v@b#S0', 'type' : 'num', 'lb' : -1, 'ub' : 1}, 
        ])
    num_uniqs = [len(space.paras[item].categories) for item in space.enum_names]
    params    = space.sample(30)
    X, Xe = space.transform(params)
    y     = torch.randn(30, 1)
    model = MaskedDeepEnsemble(X.shape[1], Xe.shape[1], 1, 
            num_uniqs  = num_uniqs,
            space      = space,
            stages     = ['S0'],
            enum_trans = enum_trans,
            num_epochs = 1)
    assert model.support_ts
    if enum_trans in ['embedding', 'onehot']:
        model.fit(X, Xe, y)
        with torch.no_grad():
            py, ps2 = model.predict(X, Xe)
            assert torch.isfinite(py).all()
            assert (ps2 > 0).all()
    else:
        with pytest.raises(RuntimeError):
            model.fit(X, Xe, y)

def train_boston_housing(params: pd.DataFrame) -> [float]:
    """data"""
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    X_trn, X_tst = FloatTensor(X_trn), FloatTensor(X_tst)
    y_trn, y_tst = FloatTensor(y_trn.reshape(-1, 1)), FloatTensor(y_tst.reshape(-1, 1))

    """models: training and predicting"""
    batch_size  = 16
    dataset     = TensorDataset(X_trn, y_trn)
    loader      = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    num_epochs  = 1
    stages      = ['Stage0', 'Stage1']

    in_features     = X_trn.shape[1]
    out_features    = 1

    target = []
    for _, param in params.iterrows():
        model   = MLP(in_features=in_features, out_features=out_features,
                      stages=stages, params=dict(param))
        """train"""
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for bx, by in loader:
                py      = model(bx)
                loss    = nn.MSELoss()(py, by)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss * batch_size

        model.eval()
        with torch.no_grad():
            py  = model(X_tst)
            err = nn.MSELoss()(py, y_tst)
        target.append(err.numpy())

    return np.array(target).reshape(-1, 1)

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, params: dict, stages: list, num_hidden: int=32):
        super(MLP, self).__init__()
        self.in_features    = in_features
        self.num_hidden     = num_hidden
        self.out_features   = out_features
        self.params         = params
        self.stages         = stages
        self.seq_layer      = self.construct_layer()

    def construct_layer(self):
        layer   = [nn.Linear(self.in_features, self.num_hidden)]
        for stage in self.stages:
            if self.params[stage] == 'RRELU':
                layer.append(nn.RReLU(lower=self.params[f'lower@RRELU#{stage}'],
                                      upper=self.params[f'upper@RRELU#{stage}']))
                layer.append(nn.Linear(self.num_hidden, self.num_hidden))
            elif self.params[stage] == 'LeakyRELU':
                layer.append(nn.LeakyReLU(negative_slope=self.params[f'negative_slope@LeakyRELU#{stage}']))
                layer.append(nn.Linear(self.num_hidden, self.num_hidden))
        layer.append(nn.Linear(self.num_hidden, self.out_features))

        return nn.Sequential(*layer)

    def forward(self, x):
        output  = self.seq_layer(x)
        return output

