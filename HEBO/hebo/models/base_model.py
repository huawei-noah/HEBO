# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import Tensor, FloatTensor, LongTensor
from .scalers import TorchStandardScaler, TorchMinMaxScaler
from abc import ABC, abstractmethod

class BaseModel(ABC):
    support_ts           = False
    support_grad         = False
    support_multi_output = False
    support_warm_start   = False

    def __init__(self, 
                 num_cont : int,
                 num_enum : int, 
                 num_out  : int, 
                 **conf):
        """
        Base class for probabilistic regression models
        conf: configuration dict
        """
        self.num_cont = num_cont
        self.num_enum = num_enum
        self.num_out  = num_out
        self.conf     = conf
        assert(self.num_cont >= 0)
        assert(self.num_enum >= 0)
        assert(self.num_out  >  0)
        assert(self.num_cont + self.num_enum > 0)
        if self.num_enum > 0:
            assert 'num_uniqs' in self.conf
            assert type(self.conf['num_uniqs']) == type([])
            assert len(self.conf['num_uniqs'])  == self.num_enum
        if not self.support_multi_output:
            assert self.num_out == 1, "Model only support single-output"

    @abstractmethod
    def fit(self, 
            Xc : FloatTensor,
            Xe : LongTensor,
            y  : FloatTensor):
        pass

    @abstractmethod
    def predict(self, 
                Xc : FloatTensor,
                Xe : LongTensor) -> (FloatTensor, FloatTensor):
        """
        Return (possibly approximated) Gaussian predictive distribution
        Return py and ps2 where py is the mean and ps2 predictive variance.
        """
        pass


    @property
    def noise(self)->FloatTensor:
        """
        Return estimated noise variance, for example, GP can view noise level
        as a hyperparameter and optimize it via MLE, another strategy could be
        using the MSE of training data as noise estimation

        Should return a (self.n_out, ) float tensor
        """
        return torch.zeros(self.num_out)

    def sample_f(self):
        # Thompson sampling
        raise NotImplementedError("Thompson sampling is not supported")

    def sample_y(self, Xc : FloatTensor, Xe : LongTensor, n_samples : int = 1) -> FloatTensor:
        py, ps2 = self.predict(Xc, Xe)
        ps      = ps2.sqrt()
        samp    = torch.zeros(n_samples, py.shape[0], self.num_out)
        for i in range(n_samples):
            samp[i] = py + ps * torch.randn(py.shape)
        return samp
