# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch 
from torch import FloatTensor, LongTensor

from .base_model import BaseModel

class MultioutputWrapper(BaseModel):
    def __init__(self,
            num_cont  : int,
            num_enum  : int,
            num_out   : int,
            model_cls : type(BaseModel),
            **conf):
        super().__init__(num_cont, num_enum, num_out)
        self.model = [model_cls(num_cont, num_enum, 1) for _ in range(self.num_out)]

    def fit(self, Xc, Xe, y):
        assert y.shape[1] == self.num_out
        for i in range(self.num_out):
            self.model[i].fit(Xc, Xe, y[:, i].view(-1, 1))

    def predict(self, Xc : FloatTensor, Xe : LongTensor) -> (FloatTensor, FloatTensor):
        py  = []
        ps2 = []
        for i in range(self.num_out):
            tmp_py, tmp_ps2 = self.model[i].predict(Xc, Xe)
            py.append(tmp_py)
            ps2.append(tmp_ps2)
        return torch.cat(py, dim = 1), torch.cat(ps2, dim = 1)

    def sample_y(self, Xc : FloatTensor, Xe : LongTensor, n_samples : int) -> FloatTensor:
        samps = [self.model[i].sample_y(Xc, Xe, n_samples) for i in range(self.num_out)]
        return torch.cat(samps, dim = -1)

    def sample_f(self):
        assert self.model[0].support_ts, "Model does not support thompson sampling"
        fs = [self.sample_f() for _ in range(self.num_out)]
        def func(Xc, Xe):
            pred = [f(Xc, Xe) for f in fs]
            return torch.cat(pred, dim = -1)
        return func

    @property
    def support_ts(self) -> bool:
        return self.model[0].support_ts

    @property
    def support_grad(self) -> bool:
        return self.model[0].support_grad

    @property
    def support_multi_output(self) -> bool:
        return True

    @property
    def support_warm_start(self) -> bool:
        return self.model[0].support_warm_start
