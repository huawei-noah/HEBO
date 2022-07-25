# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from sklearn.ensemble import RandomForestRegressor
from torch import FloatTensor, LongTensor
import numpy as np

from ..base_model import BaseModel
from ..layers import OneHotTransform
from ..util import filter_nan

class RF(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.n_estimators = self.conf.get('n_estimators', 100)
        self.rf = RandomForestRegressor(n_estimators = self.n_estimators)
        self.est_noise = torch.zeros(self.num_out)
        if self.num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])

    def xtrans(self, Xc : FloatTensor, Xe: LongTensor) -> np.ndarray:
        if self.num_enum == 0:
            return Xc.detach().numpy()
        else:
            Xe_one_hot = self.one_hot(Xe)
            if Xc is None:
                Xc = torch.zeros(Xe.shape[0], 0)
            return torch.cat([Xc, Xe_one_hot], dim = 1).numpy()

    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'all')
        Xtr = self.xtrans(Xc, Xe)
        ytr = y.numpy().reshape(-1)
        self.rf.fit(Xtr, ytr)
        mse = np.mean((self.rf.predict(Xtr).reshape(-1) - ytr)**2).reshape(self.num_out)
        self.est_noise = torch.FloatTensor(mse)

    @property
    def noise(self):
        return self.est_noise
    
    def predict(self, Xc : torch.Tensor, Xe : torch.Tensor):
        X     = self.xtrans(Xc, Xe)
        mean  = self.rf.predict(X).reshape(-1, 1)
        preds = []
        for estimator in self.rf.estimators_:
            preds.append(estimator.predict(X).reshape([-1,1]))
        var = np.var(np.concatenate(preds, axis=1), axis=1)
        return torch.FloatTensor(mean.reshape([-1,1])), torch.FloatTensor(var.reshape([-1,1])) + self.noise
