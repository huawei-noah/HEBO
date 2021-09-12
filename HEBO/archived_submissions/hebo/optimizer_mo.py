# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from sklearn.preprocessing import power_transform

import numpy  as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from bo.design_space.design_space import DesignSpace
from bo.models.gp.gp import GP
from bo.acquisitions.acq import LCB, Mean, Sigma, MOMeanSigmaLCB
from bo.optimizers.evolution_optimizer import EvolutionOpt

torch.set_num_threads(min(1, torch.get_num_threads()))

class BO(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config):
        AbstractOptimizer.__init__(self, api_config)
        self.api_config = api_config
        self.space      = self.parse_space(api_config)
        self.X          = pd.DataFrame(columns = self.space.para_names)
        self.y          = np.zeros((0, 1))
        for k in api_config:
            print(k,api_config[k])
        self.sobol      = SobolEngine(self.space.num_paras, scramble = True)

    def filter(self, y : torch.Tensor) -> [bool]:
        if not (np.all(y.numpy() > 0) and (y.max() / y.min() > 20)):
            return [True for _ in range(y.shape[0])], np.inf
        else:
            data   = y.numpy().reshape(-1)
            quant  = min(data.min() * 20, np.quantile(data, 0.95, interpolation = 'lower'))
            return (data <= quant).tolist(), quant

    def quasi_sample(self, n):
        samp    = self.sobol.draw(n)
        samp    = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x       = samp[:, :self.space.num_numeric]
        xe      = samp[:, self.space.num_numeric:]
        df_samp = self.space.inverse_transform(x, xe)
        return df_samp

    def parse_space(self, api_config):
        space  = DesignSpace()
        params = []
        for param_name in api_config:
            param_conf   = api_config[param_name]
            param_type   = param_conf['type']
            param_space  = param_conf.get('space', None)
            param_range  = param_conf.get("range", None)
            param_values = param_conf.get("values", None)

            bo_param_conf = {'name' : param_name}
            if param_type == 'int': # ignore 'log' space # TODO: support log-scale int
                bo_param_conf['type'] = 'int'
                bo_param_conf['lb']   = param_range[0]
                bo_param_conf['ub']   = param_range[1]
            elif param_type == 'bool':
                bo_param_conf['type'] = 'bool'
            elif param_type in ('cat', 'ordinal'):
                bo_param_conf['type']       = 'cat'
                bo_param_conf['categories'] = list(set(param_values))
            elif param_type == 'real':
                if param_space in ('log', 'logit'):
                    bo_param_conf['type'] = 'pow'
                    bo_param_conf['base'] = 10
                    bo_param_conf['lb']   = param_range[0]
                    bo_param_conf['ub']   = param_range[1]
                else:
                    bo_param_conf['type'] = 'num'
                    bo_param_conf['lb']   = param_range[0]
                    bo_param_conf['ub']   = param_range[1]
            else:
                assert False, "type %s not handled in API" % param_type
            params.append(bo_param_conf)
        print(params)
        space.parse(params)
        return space
            
    def suggest(self, n_suggestions=1):
        if self.X.shape[0] < 2  * self.space.num_paras:
            df_suggest = self.quasi_sample(n_suggestions)
            x_guess    = []
            for i, row in df_suggest.iterrows():
                x_guess.append(row.to_dict())
        else:
            X, Xe   = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                model = GP(self.space.num_numeric, self.space.num_categorical, 1, lr = 0.03, num_epochs = 100, verbose = True, pred_likeli = False)
                model.fit(X, Xe, y)
                print('Output scale: %g,  noise: %g %g' % (model.gp.kern.outputscale, model.lik.noise, model.noise), flush = True)
            except:
                print('Error fitting GP')
                y       = torch.FloatTensor(self.y).clone()
                filt, q = self.filter(y)
                print('Q = %g, kept = %d/%d' % (q, y.shape[0], self.y.shape[0]))
                X       = X[filt]
                Xe      = Xe[filt]
                y       = y[filt]
                model   = GP(self.space.num_numeric, self.space.num_categorical, 1, lr = 0.03, num_epochs = 100, verbose = True, pred_likeli = False, noise_lb = 8e-4)
                model.fit(X, Xe, y)


            best_id = np.argmin(self.y.squeeze())
            best_x  = self.X.iloc[[best_id]]
            best_y  = y.min()
            # best_y  = self.y.squeeze()[best_id]
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            # XXX: minimize (mu, -1 * sigma)
            #      s.t.     LCB < best_y
            iter  = max(1, self.X.shape[0] // n_suggestions)
            upsi  = 0.5
            delta = 0.05
            kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))

            acq = MOMeanSigmaLCB(model, kappa = kappa, best_y = py_best) # LCB < py_best
            mu  = Mean(model)
            sig = Sigma(model, linear_a = -1.)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 200)
            rec = opt.optimize(initial_suggest = best_x).drop_duplicates()
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = rec.append(rand_rec, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                rec      = rec.append(rand_rec, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            x_guess = []
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
                py,ps2 = model.predict(*self.space.transform(rec_selected))
                rec_selected['py'] = py.squeeze().numpy()
                rec_selected['ps'] = ps2.sqrt().squeeze().numpy()
                print(rec_selected)
            print('Best y is %g %g %g %g' % (self.y.min(), best_y, py_best, ps_best), flush = True)
            for idx in select_id:
                x_guess.append(rec.iloc[idx].to_dict())

        for rec in x_guess:
            for name in rec:
                if self.api_config[name]['type'] == 'int':
                    rec[name] = int(rec[name])
        return x_guess

    def check_unique(self, rec : pd.DataFrame) -> [bool]:
        return (~pd.concat([self.X, rec], axis = 0).duplicated().tail(rec.shape[0]).values).tolist()

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        y        = np.array(y).reshape(-1)
        valid_id = np.where(np.isfinite(y))[0].tolist()
        XX       = [X[idx] for idx in valid_id]
        yy       = y[valid_id].reshape(-1, 1)
        self.X   = self.X.append(XX, ignore_index = True)
        self.y   = np.vstack([self.y, yy])
        print(yy)


if __name__ == "__main__":
    experiment_main(BO)
