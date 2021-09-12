# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy  as np
import pandas as pd
import torch
import torch.nn as nn

from ..design_space.design_space  import DesignSpace
from ..design_space.numeric_param import NumericPara
from ..acquisitions.acq import Acquisition

class GradAcqOpt:
    def __init__(self,
            design_space : DesignSpace,
            acq          : Acquisition,
            **conf):
        self.space      = design_space
        self.acq        = acq
        self.iter       = conf.get('iters',500)
        self.verbose    = conf.get('verbose', False)
        self.lr         = conf.get('lr', 1e-1)
        self.num_obj    = self.acq.num_obj
        self.num_constr = self.acq.num_constr
        assert(self.num_obj    == 1)
        assert(self.num_constr == 0)
        for para in self.space.paras.values():
            assert para.is_numeric, "Gradient-based acquisition optimizer only supports numerical parameters"

    def callback(self):
        pass

    def optimize(self, initial_suggest : pd.DataFrame = None, num_restart = 1) -> pd.DataFrame:
        lb = self.space.opt_lb.float()
        ub = self.space.opt_ub.float()

        start_p = torch.randn(num_restart, self.space.num_paras).float()
        if initial_suggest is not None:
            init_c, _ = self.space.transform(initial_suggest.head(num_restart))
            init_c    = (torch.FloatTensor(init_c) - lb) / (ub - lb) # inverse-sigmoid transformation
            init_c    = torch.log(init_c / (1.0 - init_c + 1e-8))
            start_p[:init_c.shape[0]] = init_c
        
        best_v    = torch.tensor(np.inf)
        best_para = start_p[0].view(1, -1)

        for i in range(num_restart):
            raw_para = nn.Parameter(start_p[i].clone().view(1, -1))
            opt      = torch.optim.Adam([raw_para], lr = self.lr)
            for j in range(self.iter):
                para = torch.sigmoid(raw_para) * (ub - lb) + lb
                obj  = self.acq(para, None).squeeze()
                opt.zero_grad()
                obj.backward()
                opt.step()
            if obj.item() < best_v:
                best_para = para.detach().clone()
                best_v    = obj.item()

        with torch.no_grad():
            x      = best_para.detach()[:, :self.space.num_numeric]
            xe     = best_para.detach()[:, self.space.num_numeric:]
            df_opt = self.space.inverse_transform(x, xe)
            return df_opt
