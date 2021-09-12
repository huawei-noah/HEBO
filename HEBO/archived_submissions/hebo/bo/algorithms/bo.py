# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from ..design_space.design_space import DesignSpace
from ..

class BayesianOptimization:
    def __init__(space, func, **conf):
        self.space     = space
        self.func      = func
        self.conf      = conf
        self.rand_init = conf.get('rand_init',  1 + self.space.num_paras)
        self.max_iter  = conf.get('max_iter', 10 * self.space.num_paras)

    def optimize(self):
        df_x   = self.space.sample(self.rand_init)
        df_y   = self.func(df_x)
        assert(df_y.shape[1] == 1)
        for i in range(self.max_iter):
            model = GP(self.space.num_numeric, self.space.num_enum, 1, **self.conf)
            model.fit(*self.space.transform(df_x), torch.FloatTensor(df_y.values))
            acq     = LCB(model, kappa = 3)
            opt     = EvolutionOpt(self.space, [acq], **self.conf)
            best_id = np.argmin(df_y.values.squeeze())
            best_x  = df_x.iloc[[best_id]]
            rec_x   = opt.optimize(initial_suggest = best_x)
            rec_y   = self.func(rec_x)
            df_x    = df_x.append(rec_x, ignore_index = True)
            df_y    = df_y.append(rec_y, ignore_index = True)
