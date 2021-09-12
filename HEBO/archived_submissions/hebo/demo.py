# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from bo.design_space import DesignSpace
from bo.models.gp.gp import GP
from bo.acquisitions.lcb import LCB
from bo.optimizers import AcqOpt

def f(df_param):
    X, y = load_boston(return_X_y = True)
    tr_x, tst_x, tr_y, tst_y = train_test_split(X, y, shuffle = True, random_state = 42)
    ys  = []
    for i, row in df_param.iterrows():
        model = MLPRegressor(
                solver             = 'sgd', 
                learning_rate_init = row.lr, 
                activation         = row.activation, 
                hidden_layer_sizes = (row.num_hiddens, ), 
                momentum           = row.momentum
                ).fit(tr_x, tr_y)
        py = model.predict(tst_x)
        ys.append(r2_score(tst_y, py))
    return pd.DataFrame(ys, columns = 'obj')


if __name__ == '__main__':
    ds = DesignSpace()
    ds.parse([
        {'name' : 'lr',          'type' : 'pow', 'range': [10, -4, -2]},
        {'name' : 'momentum',    'type' : 'num', 'range': [0.1, 0.9]},
        {'name' : 'num_hiddens', 'type' : 'int', 'range': [32, 128]}, 
        {'name' : 'activation',  'type' : 'cat', 'range': ['relu', 'activation']}
    ])

    df_x = ds.sample(10)
    df_y = f(df_x)

    for i in range(40):
        xc, xe = ds.transform(df_x)
        y      = df_y.values.reshape(-1, 1)
        model  = GP(xc.shape[1], xe.shape[1])
        model.fit(xc, xe, y)
        acq     = LCB(model)
        opt     = AcqOpt(ds, [acq])
        df_rec  = opt.optimize()
        df_eval = f(df_rec)
        df_x    = df_x.append(df_rec,  ignore_index = True)
        df_y    = df_y.append(df_eval, ignore_index = True)
