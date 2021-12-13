# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from copy import deepcopy
import pandas as pd
from multiprocessing import Pool

import torch
from torch import nn
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter

from hebo.design_space.design_space import DesignSpace
from hebo.models.base_model import BaseModel
from hebo.models.scalers import TorchMinMaxScaler, TorchStandardScaler

from hebo.models.nn.eac.eac_model import EACRNN, EACTransformerEncoder, EACMLP
from hebo.models.nn.deep_ensemble import DeepEnsemble

class EACEnsemble(DeepEnsemble):
    support_ts           = True     # support Thompson sampling
    support_multi_output = True
    support_warm_start   = True
    def __init__(self,
                 num_cont: int,
                 num_enum: int,
                 num_out: int,
                 **conf):
        super(EACEnsemble, self).__init__(num_cont, num_enum, num_out, **conf)
        self.stages = self.conf.get('stages',   [])
        self.space  = self.conf.get('space',    None)
        self._check_stages()

        self.model_type = self.conf.setdefault('model_type', 'rnn')
        if self.model_type.lower() == 'rnn' or self.model_type.lower() == 'lstm':
            self.basenet_cls = EACRNN
        elif self.model_type.lower() == 'transformer':
            self.basenet_cls = EACTransformerEncoder
        elif self.model_type.lower() == 'mlp':
            self.basenet_cls = EACMLP
        else:
            raise NotImplementedError(f'{self.model_type} has not been supported.')

    def _check_stages(self):
        stages  = [name.split('#')[-1] for name in self.space.para_names if '#' in name]
        assert set(stages) == set(self.stages), \
            'Stages should be consistent to that in space'

    def fit(self, Xc: FloatTensor, Xe: LongTensor, y: FloatTensor, **fitting_conf):
        valid       = torch.isfinite(y).any(dim=1)
        X           = self.space.inverse_transform(Xc[valid], Xe[valid])
        X           = self.mask_stage(X)
        Xc_, Xe_    = self.space.transform(X)
        y_          = y[valid]

        if not self.fitted:
            self.fit_scaler(Xc=Xc_, Xe=Xe_, y=y_)
        self.yscaler.fit(y_)
        Xc_t, Xe_t, y_t = self.trans(Xc_, Xe_, y_)

        if self.num_process > 1:
            with Pool(self.num_process) as p:
                self.models = p.starmap(
                    self.fit_one,
                    [(Xc_t.clone(), Xe_t.clone(), y_t.clone(), idx)
                     for idx in range(self.num_ensembles)])
        else:
            self.models = [self.fit_one(Xc_t.clone(), Xe_t.clone(), y_t.clone(), idx)
                           for idx in range(self.num_ensembles)]

        assert None not in self.models
        self.sample_idx = 0

        with torch.no_grad():
            py, _ = self.predict(Xc_, Xe_)
            err   = py - y_
            self.noise_est = (err**2).mean(dim = 0).detach().clone()

        for i in range(self.num_ensembles):
            self.models[i].eval()

    def predict(self, Xc: FloatTensor, Xe: LongTensor) -> (FloatTensor, FloatTensor):
        X           = self.space.inverse_transform(Xc, Xe)
        X           = self.mask_stage(X)
        Xc_, Xe_    = self.space.transform(X)
        Xc_t, Xe_t  = self.trans(Xc_, Xe_)
        preds       = torch.stack([self.models[i](Xc=Xc_t, Xe=Xe_t)
                                   for i in range(self.num_ensembles)])
        if self.output_noise:
            mu      = preds[:, :, :self.num_out]
            sigma2  = preds[:, :, self.num_out:]
            py      = mu.mean(dim=0)
            ps2     = mu.var(dim=0, unbiased=False) + sigma2.mean(dim=0)
        else:
            py  = preds.mean(dim=0)
            ps2 = 1e-8 + preds.var(dim=0, unbiased=False)

        return self.yscaler.inverse_transform(FloatTensor(py)), \
               ps2 * self.yscaler.std**2

    def sample_f(self):
        assert self.fitted
        idx             = self.sample_idx
        self.sample_idx = (self.sample_idx + 1) % self.num_ensembles
        def f(Xc: FloatTensor, Xe: LongTensor) -> FloatTensor:
            model       = self.models[idx]
            X           = self.space.inverse_transform(Xc, Xe)
            X           = self.mask_stage(X)
            Xc_, Xe_    = self.space.transform(X)
            Xc_t, Xe_t  = self.trans(Xc_, Xe_)
            pred        = model(Xc_t, Xe_t)[:, :self.num_out]
            return self.yscaler.inverse_transform(pred)
        return f

    def mask_stage(self, X: pd.DataFrame) -> pd.DataFrame:
        for i, stage in enumerate(self.stages):
            stage_null  = X[stage] == 'null'
            rest_stages = self.stages[i+1:]
            X.loc[stage_null, rest_stages] = 'null'
        return X
