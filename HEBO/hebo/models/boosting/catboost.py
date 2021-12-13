# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from catboost import CatBoostRegressor, Pool, FeaturesData
from torch import FloatTensor, LongTensor
import numpy as np

from ..base_model import BaseModel
from ..util import filter_nan

class CatBoost(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.num_epochs         = self.conf.get('num_epochs', 100)   # maximum number of trees
        self.lr                 = self.conf.get('lr', 0.2)
        self.depth              = self.conf.get('depth', 10)        # recommended [1, 10]
        self.loss_function      = self.conf.get('loss_function', 'RMSEWithUncertainty')
        self.posterior_sampling = self.conf.get('posterior_sampling', True)
        self.verbose            = self.conf.get('verbose', False)
        self.random_seed        = self.conf.get('random_seed', 42)
        self.num_ensembles      = self.conf.get('num_ensembles', 10)
        if self.num_epochs < 2 * self.num_ensembles:
            self.num_epochs = self.num_ensembles * 2

        self.model = CatBoostRegressor(iterations=self.num_epochs,
                learning_rate=self.lr,
                depth=self.depth,
                loss_function=self.loss_function,
                posterior_sampling=self.posterior_sampling,
                verbose=self.verbose,
                random_seed=self.random_seed,
                allow_writing_files=False)

    def xtrans(self, Xc: FloatTensor, Xe: LongTensor) -> FeaturesData:
        num_feature_data    = Xc.numpy().astype(np.float32) if self.num_cont != 0 else None
        cat_feature_data    = Xe.numpy().astype(str).astype(object) if self.num_enum != 0 else None
        return FeaturesData(num_feature_data=num_feature_data,
                            cat_feature_data=cat_feature_data)

    def fit(self, Xc: FloatTensor, Xe: LongTensor, y: FloatTensor):
        Xc, Xe, y   = filter_nan(Xc, Xe, y, 'all')
        train_data  = Pool(data=self.xtrans(Xc=Xc, Xe=Xe), label=y.numpy().reshape(-1))
        self.model.fit(train_data)
    
    def predict(self, Xc: FloatTensor, Xe: LongTensor) -> (FloatTensor, FloatTensor):
        test_data   = Pool(data=self.xtrans(Xc=Xc, Xe=Xe))
        preds       = self.model.virtual_ensembles_predict(data=test_data,
                                                           prediction_type='TotalUncertainty',
                                                           virtual_ensembles_count=self.num_ensembles)
        mean    = preds[:, 0]
        var     = preds[:, 1] + preds[:, 2]
        return torch.FloatTensor(mean.reshape([-1,1])), \
               torch.FloatTensor(var.reshape([-1,1]))





