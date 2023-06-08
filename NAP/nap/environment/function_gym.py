# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import botorch
import gym
import gym.spaces
import numpy as np
import pandas as pd
import sobol_seq
import GPy
import json

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel

from nap.environment.util import create_uniform_grid, scale_from_unit_square_to_domain, \
    scale_from_domain_to_unit_square, get_cube_around
from nap.environment.objectives import *
from nap.policies.policies import MES
from nap.RL.utils_gp import MixtureKernel, TransformedCategorical

import pickle,random
import os
import matplotlib.pyplot as plt
import torch
import copy

class MetaBOEnv(gym.Env):
    def __init__(self, **kwargs):
        self.acqu = None
        # save shot functions
        self.shot_funcs = []
        
        self.kwargs = kwargs
        
        self.general_setting(self.kwargs["D"])


    def general_setting(self,D):
        # setting Dimension, kernel parameter, state feature
        # number of dimensions
        self.D = D
        

        # the domain (unit hypercube)
        self.domain = np.stack([np.zeros(self.D, ), np.ones(self.D, )], axis=1)

        # optimization horizon
        self.T = None  # will be set in self.reset
        if "T" in self.kwargs:
            self.T_min = self.T_max = self.kwargs["T"]
        else:
            self.T_min = self.kwargs["T_min"]
            self.T_max = self.kwargs["T_max"]
        assert self.T_min > 0
        assert self.T_min <= self.T_max

        # the initial design
        self.n_init_samples = self.kwargs["n_init_samples"]
        assert self.n_init_samples <= self.T_max
        if not (self.kwargs["f_type"] in ["HPO"]):
            self.initial_design = sobol_seq.i4_sobol_generate(self.D, self.n_init_samples)

        # the AF and its optimization
        self.af = None
        self.neg_af_and_d_neg_af_d_state = None
        self.do_local_af_opt = self.kwargs["local_af_opt"]
        if self.do_local_af_opt:
            self.N_S = self.kwargs["N_S"]
            self.discrete_domain = False

            # prepare xi_t
            self.xi_t = None  # is determined adaptively in each BO step
            self.xi_init = None
            self.af_opt_startpoints_t = None  # best k evaluations of af on multistart_grid
            self.af_maxima_t = None  # the resulting local af_maxima
            self.N_MS = self.kwargs["N_MS"]
            N_MS_per_dim = np.int(np.floor(self.N_MS ** (1 / self.D)))
            # self.multistart_grid, _ = create_uniform_grid(self.domain, N_MS_per_dim)
            self.multistart_grid = sobol_seq.i4_sobol_generate(self.D, self.N_MS)  

            self.N_MS = self.multistart_grid.shape[0]
            self.k = self.kwargs["k"]  # number of multistarts
            self.cardinality_xi_local_t = self.k
            self.cardinality_xi_global_t = self.N_S#self.N_MS

            self.cardinality_xi_t = self.cardinality_xi_local_t + self.cardinality_xi_global_t

            # hierarchical gridding or gradient-based optimization?
            self.N_LS = self.kwargs["N_LS"]
            self.local_search_grid = sobol_seq.i4_sobol_generate(self.D, self.N_LS)
            self.af_max_search_diam = 2 * 1 / N_MS_per_dim
        else:
            self.discrete_domain = True
            if self.kwargs["f_type"] == "HPO":
                self.cardinality_xi_t = self.kwargs["cardinality_domain"]
                self.xi_t = None
            else:
                self.cardinality_xi_t = self.kwargs["cardinality_domain"]
                self.xi_t = sobol_seq.i4_sobol_generate(self.D, self.kwargs["cardinality_domain"])
                self.multistart_grid = copy.deepcopy(self.xi_t)
                self.xi_init = copy.deepcopy(self.xi_t)

            self.remove_seen_points = self.kwargs.get("remove_seen_points", False)
            # will be set for once for each new function
            

        # the features
        self.features = self.kwargs["features"]
        self.feature_order_eval_envs = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"]

        # observation space
        self.n_features = 0
        if "posterior_mean" in self.features:
            self.n_features += 1
        if "posterior_std" in self.features:
            self.n_features += 1
        if "left_budget" in self.features:
            self.n_features += 1
        if "budget" in self.features:
            self.n_features += 1
        if "incumbent" in self.features:
            self.n_features += 1
        if "timestep_perc" in self.features:
            self.n_features += 1
        if "timestep" in self.features:
            self.n_features += 1
        if "x" in self.features:
            self.n_features += self.D
        if "mes" in self.features:
            self.mes = MES(dim=self.D)
            self.n_features += 1
        self.observation_space = gym.spaces.Box(low=-100000.0, high=100000.0,
                                                shape=(self.cardinality_xi_t, self.n_features),
                                                dtype=np.float32)
        self.pass_X_to_pi = self.kwargs["pass_X_to_pi"]

        # action space: index of one of the grid points
        self.action_space = gym.spaces.Discrete(self.cardinality_xi_t)

        # optimization step
        self.t = None

        # the reward
        self.reward_transformation = self.kwargs["reward_transformation"]

        # the ground truth function
        self.f_type = self.kwargs["f_type"]
        self.f_opts = self.kwargs["f_opts"]
        self.f = None
        self.y_max = None
        self.y_min = None
        self.x_max = None

        # the training data
        self.X = self.Y = None  # None means empty
        self.gp_is_empty = True

        # the surrogate GP
        self.mf = None
        self.gp = None
        self.kernel_variance = self.kwargs["kernel_variance"]
        self.kernel_lengthscale = np.array(self.kwargs["kernel_lengthscale"])
        self.noise_variance = self.kwargs["noise_variance"]
        if "use_prior_mean_function" in self.kwargs and self.kwargs["use_prior_mean_function"]:
            self.use_prior_mean_function = True
        else:
            self.use_prior_mean_function = False


    def seed(self, seed=None):
        # sets up the environment-internal random number generator and seeds it with seed
        self.rng = np.random.RandomState()
        self.seeded_with = seed
        self.rng.seed(self.seeded_with)

        if hasattr(self, "dataset_counter"):
            delattr(self, "dataset_counter")

    def set_af_functions(self, af_fun):
        # connect the policy with the environment for setting up the adaptive grid

        if not self.pass_X_to_pi:
            self.af = af_fun
        else:
            self.af = lambda state: af_fun(state, self.X, self.gp)
    def reset(self):
        if self.reward_transformation == "cumulative":
            self.cumulative_reward = 0
        if self.do_local_af_opt and not self.f_type in ["HPO"]:
            choice_indices = self.rng.choice(len(self.multistart_grid), self.N_S, replace=False)
            self.xi_init = np.array([self.multistart_grid[i] for i in choice_indices])
        elif not self.do_local_af_opt and not self.f_type in ["HPO"]:
            self.xi_t = copy.deepcopy(self.xi_init)

        # draw a new function from self.f_type
        self.draw_new_function()
        # reset the GP
        self.reset_gp()
        # reset step counters
        self.reset_step_counters()
        # optimize the AF
        self.optimize_AF()

        return self.get_state(self.xi_t)
    def setAcqu(self,acqu):
        self.acqu = acqu
    def step(self, action):
        # print(action)
        assert self.t < self.T  # if self.t == self.T one should have called self.reset() before calling this method
        if self.Y is None:
            assert self.t == 0
        else:
            assert self.t == self.Y.size - self.n_init_samples

        x_action = self.convert_idx_to_x(action)
        self.add_data(x_action)  # do this BEFORE calling get_reward()
        reward = self.get_reward(x_action)
        self.update_gp()  # do this AFTER calling get_reward()
        self.optimize_AF()
        next_state = self.get_state(self.xi_t)
        # early stop while training
        done = self.is_terminal() or ((self.t >= 30 and np.min(self.y_max-self.Y) < self.f_opts["min_regret"])
                                      and "T_max" in self.kwargs and self.f_type == "GP")

        info = {}
        if done:
            # y_diffs = (self.y_max * self.HPO_Y_std) + self.HPO_Y_mean - (self.Y * self.HPO_Y_std) + self.HPO_Y_mean
            y_diffs = self.y_max - self.Y
            simple_regret = np.asscalar(np.min(y_diffs))
            traj_regret = np.minimum.accumulate(y_diffs)
            info = {'regret': simple_regret, 'traj_regret': traj_regret, 'X': self.X, 'Y': self.Y}

        return next_state, reward, done, info

    def reset_step_counters(self):
        self.t = 0

        if self.T_min == self.T_max:
            self.T = self.T_min
        else:
            self.T = self.T_max
        assert self.T > 0  # if T was set outside of init

    def close(self):
        pass

    def draw_new_function(self):
        if "metaTrainShot" in self.f_opts:
            if 1 > len(self.shot_funcs):
                pass
            else:
                dic = self.shot_funcs[0]
                self.f = dic["f"]
                self.y_min = dic["y_min"]
                self.y_max = dic["y_max"]
                self.x_max = dic["x_max"]
                return 

        if self.f_type == "GP":
            seed = self.rng.randint(100000)
            n_features = 500
            lengthscale = self.rng.uniform(low=self.f_opts["lengthscale_low"],
                                           high=self.f_opts["lengthscale_high"])
            noise_var = self.rng.uniform(low=self.f_opts["noise_var_low"],
                                         high=self.f_opts["noise_var_high"])
            signal_var = self.rng.uniform(low=self.f_opts["signal_var_low"],
                                          high=self.f_opts["signal_var_high"])
            kernel = self.f_opts["kernel"] if "kernel" in self.f_opts else "RBF"
            # print(kernel, self.D)
            

            ssgp = SparseSpectrumGP(seed=seed, input_dim=self.D, noise_var=noise_var, 
                                    length_scale=lengthscale,
                                    signal_var=signal_var, n_features=n_features, kernel=kernel,periods=self.f_opts["periods"])
            x_train = np.array([]).reshape(0, self.D)
            y_train = np.array([]).reshape(0, 1)
            ssgp.train(x_train, y_train, n_samples=1)
            self.f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1)

            # load gp-hyperparameters
            self.kernel_lengthscale = lengthscale if not kernel == "SM" else lengthscale/4
            self.kernel_variance = signal_var
            self.noise_variance = 8.9e-16

            if self.do_local_af_opt:
                x_vec = self.xi_init
            else:
                x_vec = self.xi_t
            y_vec = self.f(x_vec)
            self.x_max = x_vec[np.argmax(y_vec)].reshape(1, self.D)
            self.y_max = np.max(y_vec)
            self.y_min = np.min(y_vec)

            self.f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1) - self.f_opts["min_regret"]

        elif self.f_type == "ackley":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: ackley_var(x, t=t, s=s)

            max_pos, max, _, min = ackley_max_min_var(self.multistart_grid,dim=self.D,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "POWELL":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: POWELL_var(x, t=t, s=s)

            max_pos, max, _, min = POWELL_max_min_var(self.multistart_grid,dim=self.D,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "egg":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: Eggholder_var(x, t=t, s=s)

            max_pos, max, _, min = Eggholder_max_min_var(self.multistart_grid,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "GRIEWANK":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: GRIEWANK_var(x, t=t, s=s)

            max_pos, max, _, min = GRIEWANK_max_min_var(self.multistart_grid,self.D,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "DIXON_PRICE":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: DIXON_PRICE_var(x, t=t, s=s)

            max_pos, max, _, min = DIXON_PRICE_max_min_var(self.multistart_grid,self.D,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "STYBLINSKI_TANG":
            if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, self.D))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
            else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: STYBLINSKI_TANG_var(x, t=t, s=s)

            max_pos, max, _, min = STYBLINSKI_TANG_max_min_var(self.multistart_grid,self.D,t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = -5
        elif self.f_type == "HPO": 

            if not hasattr(self, "dataset_counter"):
                self.dataset_counter = self.rng.randint(len(self.kwargs["f_opts"]["data"]))
            else:
                self.dataset_counter += 1
            if self.dataset_counter >= len(self.kwargs["f_opts"]["data"]):
                self.dataset_counter = 0

            self.pkl_data = pickle.load(open(self.kwargs["f_opts"]["data"][self.dataset_counter],"rb"))

            if self.kwargs["f_opts"].get("shuffle_and_cutoff", False):
                # shuffle and truncate to get always same size datasets
                random_index = np.arange(len(self.pkl_data["accs"]))
                self.rng.shuffle(random_index)
                random_index_trunc = random_index[:self.kwargs.get("cardinality_domain")]
                self.pkl_data["domain"] = self.pkl_data["domain"][random_index_trunc]
                self.pkl_data["accs"] = self.pkl_data["accs"][random_index_trunc]
            
            if self.kwargs["f_opts"].get("y_norm", False):
                self.pkl_data["accs"] = (self.pkl_data["accs"] - self.pkl_data["accs"].min()) / (self.pkl_data["accs"].max() - self.pkl_data["accs"].min())
            if self.kwargs["f_opts"].get("x_norm", False):
                xrange = self.pkl_data["domain"].max(0) - self.pkl_data["domain"].min(0)
                xrange = np.where(xrange == 0, 1, xrange)
                self.pkl_data["domain"] = (self.pkl_data["domain"] - self.pkl_data["domain"].min(0)) / xrange

            self.xi_t = get_HPO_domain(data=self.pkl_data)

            self.cardinality_xi_t = len(self.xi_t)
            initial = self.rng.choice(np.arange(self.cardinality_xi_t),size=self.n_init_samples,replace=False)
            self.initial_design = []
            for init in initial:
                if self.kwargs.get('use_index_speedup', False):
                    self.initial_design.append(init)
                else:
                    self.initial_design.append(self.convert_idx_to_x(init))
            self.initial_design = np.array(self.initial_design)
            assert self.xi_t.shape[0] == self.cardinality_xi_t

            self.f = lambda x, idx=None: HPO(x, data=self.pkl_data, index=idx)

            max_pos, max, _, min = HPO_max_min(data=self.pkl_data)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min

        elif self.f_type == "Antigen":

            if not hasattr(self, "dataset_counter"):
                self.dataset_counter = self.rng.randint(len(self.kwargs["f_opts"]["data"]))
            else:
                self.dataset_counter += 1
            if self.dataset_counter >= len(self.kwargs["f_opts"]["data"]):
                self.dataset_counter = 0

            def clean(seq_string):
                return list(map(int, seq_string.split(',')))

            # self.pkl_data = pickle.load(open(self.kwargs["f_opts"]["data"][self.dataset_counter], "rb"))
            df = pd.read_csv(self.kwargs["f_opts"]["data"][self.dataset_counter], converters={'domain': clean})
            tokenized_seq = df['domain'].values
            X = np.stack(tokenized_seq)
            Y = df["accs"].values
            normY = (Y - Y.min()) / (Y.max() - Y.min())
            self.pkl_data = {"domain": X, "accs": normY}

            if self.kwargs["f_opts"].get("shuffle_and_cutoff", False):
                # shuffle and truncate to get always same size datasets
                random_index = np.arange(len(self.pkl_data["accs"]))
                self.rng.shuffle(random_index)
                random_index_trunc = random_index[:self.kwargs.get("cardinality_domain")]
                self.pkl_data["domain"] = self.pkl_data["domain"][random_index_trunc]
                self.pkl_data["accs"] = self.pkl_data["accs"][random_index_trunc]

            if self.f_opts.get("y_std", False):
                self.Antigen_Y_mean = self.pkl_data["accs"].mean()
                self.Antigen_Y_std = self.pkl_data["accs"].std()
                self.pkl_data["accs"] = (self.pkl_data["accs"] - self.Antigen_Y_mean) / self.Antigen_Y_std
            if self.f_opts.get("x_std", False):
                self.Antigen_X_mean = self.pkl_data["domain"].mean(0)
                self.Antigen_X_std = self.pkl_data["domain"].std(0)
                self.Antigen_X_mean[-5:] = 0.  # assumes categorical dimensions are the 5 last ones
                self.Antigen_X_std[-5:] = 1.

            self.xi_t = self.pkl_data["domain"]
            self.cardinality_xi_t = len(self.xi_t)
            initial = self.rng.choice(np.arange(self.cardinality_xi_t), size=self.n_init_samples, replace=False)
            self.initial_design = []
            for init in initial:
                if self.kwargs.get('use_index_speedup', False):
                    self.initial_design.append(init)
                else:
                    self.initial_design.append(self.convert_idx_to_x(init))
            self.initial_design = np.array(self.initial_design)
            assert self.xi_t.shape[0] == self.cardinality_xi_t

            max_pos, max, _, min = Antigen_max_min(data=self.pkl_data)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min

            self.f = lambda x, idx=None: Antigen(x, data=self.pkl_data, index=idx)

        else:
            raise ValueError("Unknown f_type!")

        if "metaTrainShot" in self.f_opts:
            if 1 > len(self.shot_funcs):
                dic = {"f":copy.deepcopy(self.f),
                        "x_max":self.x_max,
                        "y_max":self.y_max,
                        "y_min":self.y_min}
                self.shot_funcs.append(dic)

        assert self.y_max is not None  # we need this for the reward
        assert self.y_min is not None  # we need this for the incumbent of empty training set

    def reset_gp(self):
        if self.kwargs.get('random_init_gp', False):
            self.noise_variance = np.random.uniform(0.1, 1.0)
            self.kernel_lengthscale = np.random.uniform(0.5, 1.0)
            self.kernel_variance = 1.0

        # reset gp
        if "kernel" in self.f_opts and "Matern52" in self.f_opts["kernel"]:
            self.kernel = GPy.kern.Matern52(input_dim=self.D,
                                variance=self.kernel_variance,
                                lengthscale=self.kernel_lengthscale,
                                ARD=True)
        elif "kernel" in self.f_opts and "Matern" in self.f_opts["kernel"]:
            self.kernel = GPy.kern.Matern32(input_dim=self.D,
                                variance=self.kernel_variance,
                                lengthscale=self.kernel_lengthscale,
                                ARD=True)
        elif "kernel" in self.f_opts and "ExpSS" in self.f_opts["kernel"]:
            self.kernel = GPy.kern.StdPeriodic(input_dim=self.D,
                                    variance=self.kernel_variance,
                                    lengthscale=np.ones((self.D))*self.kernel_lengthscale,
                                    period=0.5,
                                    ARD2=True)
        elif "kernel" in self.f_opts and "Mixture" in self.f_opts["kernel"]:
            cat_dims = np.array(self.f_opts["cat_dims"])
            cont_dims = np.array(self.f_opts["cont_dims"])
            cat_dims += len(cat_dims) + len(cont_dims) if np.all(cat_dims < 0) else 0
            self.kernel = MixtureKernel(
                categorical_dims=cat_dims,
                continuous_dims=cont_dims,
            )
            self.kernel.continuous_kern.lengthscale = self.f_opts["continuous_kern_lengthscale"]
            self.kernel.categorical_kern.lengthscale = self.f_opts["categorical_kern_lengthscale"]
            self.kernel.lengthscale = self.f_opts["outputscale"]
            self.kernel.lamda = self.f_opts["lamda"]
        elif "kernel" in self.f_opts and self.f_opts["kernel"] == 'Categorical':
            cat_dims = np.array(self.f_opts["cat_dims"])
            self.kernel = ScaleKernel(TransformedCategorical(
                ard_num_dims=len(cat_dims)
            ))
            self.kernel.base_kernel.lengthscale = self.f_opts["lengthscale"]
        else:
            self.kernel = GPy.kern.RBF(input_dim=self.D,
                                variance=self.kernel_variance,
                                lengthscale=self.kernel_lengthscale,
                                ARD=True)

        if self.use_prior_mean_function:
            self.mf = GPy.core.Mapping(self.D, 1)
            self.mf.f = lambda X: np.mean(self.Y, axis=0)[0] if self.Y is not None else 0.0
            self.mf.update_gradients = lambda a, b: 0
            self.mf.gradients_X = lambda a, b: 0
        else:
            self.mf = None

        normalizer = False

        # this is only dummy data as GPy is not able to have empty training set
        # for prediction, the GP is not used for empty training set
        if self.n_init_samples > 0:
            X = self.initial_design.reshape(self.n_init_samples,self.D)
            Y = []
            for inid in self.initial_design:
                Y.append(self.f(inid.reshape(1,-1)))
            Y = np.array(Y).reshape(self.n_init_samples,1)
            self.X = X
            self.Y = Y
        else:
            self.X = self.Y = None
            X = np.zeros((1, self.D))
            Y = np.zeros((1, 1))
        self.gp_is_empty = True

        if isinstance(self.kernel, MixtureKernel):
            self.gp = SingleTaskGP(train_X=torch.from_numpy(X),
                                   train_Y=torch.from_numpy(Y).view(-1, 1),
                                   covar_module=self.kernel)
            self.gp.covar_module.continuous_kern.lengthscale = self.f_opts["continuous_kern_lengthscale"]
            self.gp.covar_module.categorical_kern.lengthscale = self.f_opts["categorical_kern_lengthscale"]
            self.gp.covar_module.lengthscale = self.f_opts["outputscale"]
            self.gp.covar_module.lamda = self.f_opts["lamda"]
            self.gp.mean_module.constant = self.f_opts["mean_constant"]
            self.gp.likelihood.noise = self.f_opts["likelihood_noise"]
        elif isinstance(self.kernel.base_kernel, TransformedCategorical):
            self.gp = SingleTaskGP(train_X=torch.from_numpy(X).to(float),
                                   train_Y=torch.from_numpy(Y),
                                   covar_module=self.kernel)
            self.gp.covar_module.base_kernel.lengthscale = self.f_opts['lengthscale']
            self.gp.likelihood.noise = self.f_opts['likelihood_noise']
            self.gp.mean_module.constant = self.f_opts['mean_constant']
        else:
            self.gp = GPy.models.gp_regression.GPRegression(X, Y,
                                                            noise_var=self.noise_variance,
                                                            kernel=self.kernel,
                                                            mean_function=self.mf,
                                                            normalizer=normalizer)
            self.gp.Gaussian_noise.variance = self.noise_variance
            self.gp.kern.lengthscale = self.kernel_lengthscale
            self.gp.kern.variance = self.kernel_variance
    def add_data(self, X):
        assert X.ndim == 2

        # evaluate f at X and add the result to the GP
        Y = self.f(X)

        if not self.do_local_af_opt and self.remove_seen_points:
            self.xi_t = self.xi_t[~((self.xi_t == X).all(-1))]
            self.cardinality_xi_t = len(self.xi_t)

        if self.X is None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)

        self.t += X.shape[0]

    def update_gp(self):
        assert self.Y is not None
        if isinstance(self.kernel, MixtureKernel):
            previous_cont_ls = self.gp.covar_module.continuous_kern.lengthscale.clone().detach()
            previous_cat_ls = self.gp.covar_module.categorical_kern.lengthscale.clone().detach()
            previous_cov_ls = self.gp.covar_module.lengthscale.clone().detach()
            previous_lamda = self.gp.covar_module.lamda
            previous_m = self.gp.mean_module.constant.clone().detach()
            previous_ns = self.gp.likelihood.noise.clone().detach()
            self.gp = SingleTaskGP(train_X=torch.from_numpy(self.X),
                                   train_Y=torch.from_numpy(self.Y).view(-1, 1),
                                   covar_module=self.kernel)
            self.gp.covar_module.continuous_kern.lengthscale = previous_cont_ls
            self.gp.covar_module.categorical_kern.lengthscale = previous_cat_ls
            self.gp.covar_module.lengthscale = previous_cov_ls
            self.gp.covar_module.lamda = previous_lamda
            self.gp.mean_module.constant = previous_m
            self.gp.likelihood.noise = previous_ns
            if self.kwargs.get('update_gp', False) and self.X is not None and self.X.shape[0] > 1:
                try:
                    mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
                    _ = fit_gpytorch_mll(mll=mll)
                except botorch.exceptions.errors.ModelFittingError as e:
                    print(e)
                    mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
                    _ = fit_gpytorch_mll_torch(mll=mll)
                    print('fallback cuda training completed')
                mll.eval()
        elif isinstance(self.kernel.base_kernel, TransformedCategorical):
            previous_ls = self.gp.covar_module.base_kernel.lengthscale.clone().detach()
            previous_os = self.gp.covar_module.outputscale.clone().detach()
            previous_ns = self.gp.likelihood.noise.clone().detach()
            previous_m = self.gp.mean_module.constant.clone().detach()
            self.gp = SingleTaskGP(train_X=torch.from_numpy(self.X).to(float),
                                   train_Y=torch.from_numpy(self.Y),
                                   covar_module=self.kernel)
            self.gp.covar_module.base_kernel.lengthscale = previous_ls
            self.gp.covar_module.outputscale = previous_os
            self.gp.likelihood.noise = previous_ns
            self.gp.mean_module.constant = previous_m
            if self.kwargs.get('update_gp', False) and self.X is not None and self.X.shape[0] > 1:
                try:
                    mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
                    _ = fit_gpytorch_mll(mll=mll)
                except botorch.exceptions.errors.ModelFittingError as e:
                    print(e)
                    mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
                    _ = fit_gpytorch_mll_torch(mll=mll)
                    print('fallback cuda training completed')
                mll.eval()
        else:
            self.gp.set_XY(self.X, self.Y)
            if self.kwargs.get('update_gp', False) and self.X is not None and self.X.shape[0] > 1:
                ret = self.gp.optimize(messages=False, ipython_notebook=False)
                if ret.status != "Converged":
                    self.gp.Gaussian_noise.variance = self.noise_variance
                    self.gp.kern.lengthscale = self.kernel_lengthscale
                    self.gp.kern.variance = self.kernel_variance
                elif np.isnan(self.gp.Gaussian_noise.variance) or np.isnan(self.gp.kern.lengthscale).any() or np.isnan(self.gp.kern.variance) or \
                    np.abs(self.gp.Gaussian_noise.variance) > 100000 or (np.abs(self.gp.kern.lengthscale) > 100000).any() or np.abs(self.gp.kern.variance) > 100000:
                    self.gp.Gaussian_noise.variance = self.noise_variance
                    self.gp.kern.lengthscale = self.kernel_lengthscale
                    self.gp.kern.variance = self.kernel_variance

        self.gp_is_empty = False

    def optimize_AF(self):
        if self.do_local_af_opt:
            # obtain maxima of af
            self.get_af_maxima()
            self.xi_t = np.concatenate((self.af_maxima_t, self.xi_init), axis=0)
            assert self.xi_t.shape[0] == self.cardinality_xi_t
        else:
            pass

    def get_state(self, X):
        # fill the state
        self.rng.shuffle(X)
        feature_count = 0
        idx = 0
        state = np.zeros((X.shape[0], self.n_features), dtype=np.float32)
        gp_mean, gp_std = self.eval_gp(X)
        if "posterior_mean" in self.features:
            feature_count += 1
            state[:, idx:idx + 1] = gp_mean.reshape(X.shape[0], 1)
            idx += 1
        if "posterior_std" in self.features:
            feature_count += 1
            state[:, idx:idx + 1] = gp_std.reshape(X.shape[0], 1)
            idx += 1
        if "x" in self.features:
            feature_count += 1
            state[:, idx:idx + self.D] = X
            idx += self.D
        if "incumbent" in self.features:
            feature_count += 1
            incumbent_vec = np.ones((X.shape[0],)) * self.get_incumbent()
            state[:, idx] = incumbent_vec
            idx += 1
        if "timestep_perc" in self.features:
            feature_count += 1
            t_perc = self.t / self.T
            t_perc_vec = np.ones((X.shape[0],)) * t_perc
            state[:, idx] = t_perc_vec
            idx += 1
        if "timestep" in self.features:
            feature_count += 1
            # clip timestep
            if "T_training" in self.kwargs and self.kwargs["T_training"] is not None:
                t = np.min([self.t, self.kwargs["T_training"]])
            else:
                t = self.t
            t_vec = np.ones((X.shape[0],)) * t
            state[:, idx] = t_vec
            idx += 1
        if "budget" in self.features:
            feature_count += 1
            if "T_training" in self.kwargs and self.kwargs["T_training"] is not None:
                T = self.kwargs["T_training"]
            else:
                T = self.T
            budget_vec = np.ones((X.shape[0],)) * T
            state[:, idx] = budget_vec
            idx += 1
        if "mes" in self.features:
            feature_count += 1
            state[:, idx:idx + 1] = self.mes.af(X,None,self.gp).reshape(X.shape[0], 1)
            idx += 1
        if "mes_y*" in self.features:
            feature_count += 1
            self.mes.gp = self.gp
            state[:, idx:idx] = np.ones((X.shape[0],))*self.mes._sample_maxes()
            idx += 1
        if "mu*" in self.features:
            feature_count += 1
            state[:, idx:idx] = np.ones((X.shape[0],)) * np.max(self.af(state))
            idx += 1

        assert idx == self.n_features  # make sure the full state has been filled
        if not feature_count == len(self.features):
            raise ValueError("Invalid feature specification!")

        return state

    def get_reward(self,X):
        # make sure you already increased the step counter self.t before calling this method!
        # make sure you updated the training set but did NOT update the gp before calling this method!
        assert self.Y is not None  # this method should not be called with empty training set
        negativity_check = False

        # compute the simple regret
        y_diffs = self.y_max - self.Y
        simple_regret = np.min(y_diffs)
        reward = np.asscalar(simple_regret)

        # apply reward transformation
        if self.reward_transformation == "none":
            reward = reward
        elif self.reward_transformation == "cumulative":
            y_diffs = self.y_max - self.f(X)
            simple_regret = np.min(y_diffs)
            reward = np.asscalar(simple_regret)
            reward = np.max((1e-20, reward))
            self.cumulative_reward += reward
            return self.cumulative_reward
        elif self.reward_transformation == "neg_linear":
            reward = -reward
        elif self.reward_transformation == "neg_log10":
            # if reward < 1e-20:
            #     print("Warning: logarithmic reward may be invalid!")
            reward, negativity_check = np.max((1e-20, reward)), True
            assert negativity_check
            reward = -np.log10(reward)
        else:
            raise ValueError("Unknown reward transformation!")

        return reward

    def get_af_maxima(self):
        state_at_multistarts = self.get_state(self.xi_init)### shape(961,6)
        af_at_multistarts = self.af(state_at_multistarts)
        self.af_opt_startpoints_t = self.xi_init[np.argsort(-af_at_multistarts)[:self.k, ...]]
        local_grids = [scale_from_unit_square_to_domain(self.local_search_grid,
                                                        domain=get_cube_around(x,
                                                                               diam=self.af_max_search_diam,
                                                                               domain=self.domain))
                       for x in self.af_opt_startpoints_t]
        local_grids = np.concatenate(local_grids, axis=0)
        state_on_local_grid = self.get_state(local_grids)#### shape(5000,6)
        af_on_local_grid = self.af(state_on_local_grid)
        self.af_maxima_t = local_grids[np.argsort(-af_on_local_grid)[:self.cardinality_xi_local_t]]### choose the last 5 point
        assert self.af_maxima_t.shape[0] == self.cardinality_xi_local_t

    def get_incumbent(self):
        if self.Y is None:
            Y = np.array([self.y_min])
        else:
            Y = self.Y

        incumbent = np.max(Y)
        return incumbent

    def eval_gp(self, X_star):
        # evaluate the GP on X_star
        assert X_star.shape[1] == self.D
        if isinstance(self.kernel, MixtureKernel):
            if self.gp_is_empty:
                gp_mean = np.zeros((X_star.shape[0],))
                gp_std = np.sqrt(1 * np.ones((X_star.shape[0],)))
            else:
                gp_posterior = self.gp.posterior(torch.from_numpy(X_star))
                gp_mean = gp_posterior.mean.detach().cpu().numpy()
                gp_std = gp_posterior.variance.sqrt().detach().cpu().numpy()
        elif isinstance(self.kernel.base_kernel, TransformedCategorical):
            if self.gp_is_empty:
                gp_mean = np.zeros((X_star.shape[0],))
                gp_std = np.sqrt(1 * np.ones((X_star.shape[0],)))
            else:
                gp_posterior = self.gp.posterior(torch.from_numpy(X_star).to(float))
                gp_mean = gp_posterior.mean.detach().cpu().numpy()
                gp_std = gp_posterior.variance.sqrt().detach().cpu().numpy()
        else:
            if self.gp_is_empty:
                gp_mean = np.zeros((X_star.shape[0],))
                gp_var = self.kernel_variance * np.ones((X_star.shape[0],))
            else:
                gp_mean, gp_var = self.gp.predict_noiseless(X_star)
                gp_mean = gp_mean[:, 0]
                gp_var = gp_var[:, 0]
            gp_std = np.sqrt(gp_var)

        return gp_mean, gp_std

    def neg_af(self, x):
        x = x.reshape(1, self.D)  # the optimizer queries one point at a time
        state = self.get_state(x)
        neg_af = -self.af(state)

        return neg_af

    def get_random_sampling_reward(self):
        self.reset_step_counters()
        self.draw_new_function()

        self.X, self.Y = None, None
        rewards = []
        for t in range(self.T):
            if not self.discrete_domain:
                random_sample = self.rng.rand(1, self.D)
            else:
                random_sample = self.xi_t[self.rng.choice(np.arange(self.cardinality_xi_t)), :].reshape(1, -1)
            self.X = np.concatenate((self.X, random_sample), axis=0) if self.X is not None else random_sample
            f_x = self.f(random_sample)
            self.Y = np.concatenate((self.Y, f_x), axis=0) if self.Y is not None else f_x
            rewards.append(self.get_reward(self.X))
            self.t += 1

        assert self.is_terminal()

        return rewards

    def convert_idx_to_x(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array([idx])
        return self.xi_t[idx, :].reshape(idx.size, self.D)

    def is_terminal(self):
        return self.t == self.T
