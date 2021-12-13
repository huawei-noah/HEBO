# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from copy import  deepcopy
from multiprocessing import Pool
import pandas as pd

import torch
from torch import nn
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset

from hebo.design_space.design_space import DesignSpace
from hebo.models.base_model import BaseModel
from hebo.design_space.bool_param import BoolPara
from hebo.design_space.categorical_param import CategoricalPara


class ConditionalDeepEnsemble(BaseModel):
    support_multi_output = True
    support_warm_start   = True
    def __init__(self,
                 num_cout: int,
                 num_enum: int,
                 num_out: int=1,
                 space: DesignSpace=None,
                 conditions: dict=None,
                 **conf):
        """
        DeepEnsemble model with conditional design space where we know some
        parameters are enabled/disabled by other parameters

        Parameters
        -----------------------
        num_cont: int, number of numerical parameters
        num_enum: int, number of categorical parameters
        num_out: int, number of output
        space : BO design space
        conditions : dict, conditional relationships

        Example
        -----------------------
        For example, we have [param_A, param_B, param_C , parma_D ] to model,
        where `param_A` is a boolean parameter and other three parameters are
        all continuous parameters ranging from 0 to 1. We want to optimize
        `param_B` and `param_C` if `param_A == True`, otherwise we only want to
        optimize `param_D`

        Then the mode can be initialized like

        >>>space = DesignSpace().parse([{
                        {'name' : 'param_A', type : 'bool'}, 
                        {'name' : 'param_B', type : 'num', 'lb' : 0, 'ub' : 1}, 
                        {'name' : 'param_C', type : 'num', 'lb' : 0, 'ub' : 1}, 
                        {'name' : 'param_D', type : 'num', 'lb' : 0, 'ub' : 1}
                   ]})
        >>>cond = {'param_A' : None, 
                   'parma_B' : ('param_A', True),  
                   'parma_C' : ('param_A', True),  
                   'parma_D' : ('param_A', False), 
                   }
        >>>model = ConditionalDeepEnsemble(4, 0, 1,
                                           num_uniqs = None,
                                           param_names = param_names,
                                           conditions = cond)
        """
        super(ConditionalDeepEnsemble, self).__init__(num_cout, num_enum, num_out, **conf)
        self.space          = space
        self.conditions     = self._check_conditions(conditions)
        self.num_out        = num_out
        self.conf           = conf
        self.output_noise   = self.conf.get('output_noise', True)
        self.num_ensemble   = self.conf.get('num_ensembles', 5)
        self.num_process    = self.conf.get('num_processes', 1)
        self.num_epoch      = self.conf.get('num_epoch', 500)
        self.print_every    = self.conf.get('print_every', 50)

        self.num_layer  = self.conf.get('num_layer', 1)
        self.num_hidden = self.conf.get('num_hidden', 64)
        self.batch_size = self.conf.get('batch_size', 16)
        self.l1         = self.conf.get('l1', 1e-3)
        self.lr         = self.conf.get('lr', 5e-3)
        self.verbose    = self.conf.get('verbose', True)

        self.loss       = self.loss_likelihood if self.output_noise else self.loss_mse
        self.loss_name  = 'NLL' if self.output_noise else 'MSE'
        self.models     = None

    def loss_mse(self, pred, target):
        mask = torch.isfinite(target)
        return nn.MSELoss()(pred[mask], target[mask])

    def loss_likelihood(self, pred, target):
        mask = torch.isfinite(target)
        mu = pred[:, :self.num_out][mask]
        sigma2 = pred[:, self.num_out:][mask]
        loss = 0.5 * (target[mask] - mu)**2 / sigma2 + 0.5 * torch.log(sigma2)
        return torch.mean(loss)

    def _check_conditions(self, conditions):
        conditions_new = {}
        components = [k for k, v in conditions.items() if v is None]
        for k, v in conditions.items():
            if v is not None:
                if v[0] in set(components):
                    if not isinstance(v[1], list):
                        conditions_new[k] = (v[0], [v[1]])
                    else:
                        conditions_new[k] = (v[0], v[1])
                else:
                    raise ValueError('%s is not valid!' % v[0])
            else:
                conditions_new[k] = None
        return conditions_new

    @property
    def components(self):
        _components = [k for k, v in self.conditions.items() if v is None]
        res = {}
        for _c in _components:
            if isinstance(self.space.paras[_c], BoolPara):
                res[_c] = {True: [], False: []}
            elif isinstance(self.space.paras[_c], CategoricalPara):
                res[_c] = {c:[] for c in self.space.paras[_c].categories}
            else:
                raise NotImplementedError('Out of Bool and Categorical!')

        for k, v in self.conditions.items():
            if v is not None:
                for _v in v[1]:
                    res[v[0]][_v].append(k)

        return res

    @property
    def partitions(self):
        _partitions = []
        for swither, params in self.components.items():
            partition = [swither]
            for p in params.values():
                partition += p
            _partitions.append(list(set(partition)))
        return _partitions

    @property
    def in_columns(self):
        cols = ['%s@%s@%s' % (p, switcher, component)
                for component, v in self.components.items()
                for switcher, params in v.items()
                for p in params]
        return cols

    @property
    def in_splits(self):
        splits = []
        for component, v in self.components.items():
            cols = [col for col in self.in_columns
                    if col.split('@')[-1] == component]
            splits.append(cols)
        return [len(cols) for cols in splits]

    def data_processing(self, X: pd.DataFrame, y: pd.DataFrame=None):
        assert X.shape[1] == sum([len(p) for p in self.partitions])

        Xs = [X[p] for p in self.partitions]
        Xs_processed = []
        for i, (component, v) in enumerate(self.components.items()):
            _Xs = []
            for switcher, params in v.items():
                _columns = {p: '%s@%s@%s' % (p, switcher, component) for p in params}
                _X = Xs[i].loc[Xs[i][component] == switcher, params].rename(columns=_columns)
                _Xs.append(pd.concat([Xs[i].loc[Xs[i][component] == switcher, [component]], _X], axis=1))
            Xs_processed.append(pd.concat(_Xs, axis=0).fillna(0))
        df_reorder = pd.concat(Xs_processed, axis=1).sort_index()
        X_tensor = torch.tensor(df_reorder[self.in_columns].values)
        ## pd.concat reorder the index of data
        return X_tensor if y is None else (X_tensor, torch.tensor(y.values))

    def fit(self,
            Xc : FloatTensor,
            Xe : LongTensor,
            y  : FloatTensor):
        X = self.space.inverse_transform(Xc, Xe)
        y = pd.DataFrame(y.numpy())

        if self.num_process > 1:
            with Pool(self.num_process) as p:
                self.models = p.starmap(self.fit_one,
                                        [(deepcopy(X), deepcopy(y), idx)
                                         for idx in range(self.num_ensemble)])
        else:
            self.models = [self.fit_one(X, y, idx)
                           for idx in range(self.num_ensemble)]

        assert None not in self.models


    def predict(self,
                Xc : FloatTensor,
                Xe : LongTensor) -> (FloatTensor, FloatTensor):
        X = self.space.inverse_transform(Xc, Xe)
        X = self.data_processing(X=X)
        preds = torch.stack([self.models[i](X) for i in range(self.num_ensemble)])
        if not self.output_noise:
            py = preds.mean(dim=0)
            ps2 = 1e-8 + preds.var(dim= 0, unbiased=False) # XXX: var([1.0], unbiased = True) = NaN
        else:
            mu = preds[:, :, :self.num_out]
            sigma2 = preds[:, :, self.num_out:]
            py = mu.mean(dim=0)
            ps2 = mu.var(dim=0, unbiased=False) + sigma2.mean(dim=0)
        return py, ps2

    def fit_one(self, X: pd.DataFrame, y: pd.DataFrame, idx: int):
        torch.seed()
        self.X, self.y = self.data_processing(X=X, y=y)
        assert self.in_splits is not None

        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.models is not None and len(self.models) == self.num_ensemble:
            model = deepcopy(self.models[idx])
        else:
            model = ConditionalNet(in_splits=self.in_splits,
                                   num_out=self.num_out,
                                   output_noise=self.output_noise)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            for bx, by in dataloader:
                py = model(bx)
                data_loss = self.loss(py, by)
                reg_loss  = 0.
                for p in model.parameters():
                    reg_loss += self.l1 * p.abs().sum() / (y.shape[0] * y.shape[1])
                loss = data_loss + reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                # XXX: Adversarial training not used

                epoch_loss += data_loss * bx.shape[0]
            if epoch % self.print_every == 0:
                if self.verbose:
                    print("Epoch %d, %s loss = %g" %
                          (epoch, self.loss_name, epoch_loss / X.shape[0]),
                          flush = True)
        return model



class ConditionalNet(nn.Module):
    def __init__(self, in_splits: list, num_out: int=1, **conf):
        super(ConditionalNet, self).__init__()
        self.num_out = num_out
        self.in_splits = in_splits
        self.num_components = len(in_splits)
        self.num_hidden = conf.get('num_hidden', 48)
        self.output_noise = conf.get('output_noise', True)

        self.component_layer = self.construct_component_layer()
        self.common_layer = self.construct_common_layer()
        self.mu = nn.Linear(self.num_hidden, self.num_out)
        if self.output_noise:
            self.sigma2 = nn.Sequential(
                nn.Linear(self.num_hidden, self.num_out),
                nn.Softplus()
            )

    def construct_component_layer(self):
        componet_layer = nn.ModuleList([])
        for input_size in self.in_splits:
            componet_layer.append(nn.Sequential(
                nn.Linear(input_size, self.num_hidden),
                nn.BatchNorm1d(self.num_hidden),
                nn.ReLU()
            ))
        return componet_layer

    def construct_common_layer(self):
        common_layer = nn.Sequential(
            nn.Linear(self.num_hidden * self.num_components, self.num_hidden),
            nn.BatchNorm1d(self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.BatchNorm1d(self.num_hidden),
            nn.ReLU()
        )
        return common_layer

    def forward(self, X: FloatTensor):
        # assert X.shape[1] == len([j for s in self.in_splits for j in s])
        assert X.shape[1] == sum(self.in_splits)
        assert len(self.in_splits) == len(self.component_layer)

        output = [component(x)
                  for x, component in zip(X.split(self.in_splits, dim=1),
                                          self.component_layer)]
        output = torch.cat(output, dim=1)
        output = self.common_layer(output)
        mu = self.mu(output)
        output = torch.cat([mu, self.sigma2(output)], dim=1) if self.output_noise else mu
        return output

# if __name__ == '__main__':
#     import pandas as pd
#     from hebo.design_space.design_space import DesignSpace

#     space = DesignSpace().parse([
#         {'name': 'stage1', 'type': 'cat', 'categories': ['alg1', 'alg2']},
#         {'name': 'stage2', 'type': 'cat', 'categories': ['alg3', 'alg4']},
#         {'name': 'p1', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p2', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p3', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p4', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p5', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p6', 'type': 'num', 'lb': 0, 'ub': 1},
#         {'name': 'p7', 'type': 'num', 'lb': 0, 'ub': 1},
#     ])

#     X = space.sample(20)
#     y = torch.randn(20, 2)

#     cond = {'stage1': None,
#             'stage2': None,
#             'p1': ('stage1', ['alg1', 'alg2']),
#             'p2': ('stage1', ['alg1', 'alg2']),
#             'p3': ('stage1', ['alg1', 'alg2']),
#             'p4': ('stage2', ['alg3']),
#             'p5': ('stage2', ['alg3', 'alg4']),
#             'p6': ('stage2', ['alg3']),
#             'p7': ('stage2', ['alg4']),
#             }

#     model = ConditionalDeepEnsemble(num_cout=7, num_enum=2, num_out=2,
#                                     conditions=cond,
#                                     space=space,
#                                     num_epoch=10,
#                                     num_uniqs=[2, 2])
#     # model = ConditionalDeepEnsemble(conditions=cond, num_processes=5)     # need to be run in another main such as demo
#     model.fit(*space.transform(X), y)

#     py, ps2 = model.predict(*space.transform(X))
