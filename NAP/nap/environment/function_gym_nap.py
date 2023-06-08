# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import gym
import gym.spaces
import pandas as pd
import pickle
import torch
import copy

from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from nap.environment.util import scale_from_unit_square_to_domain, get_cube_around
from nap.environment.objectives import *

class NAPEnv(gym.Env):
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
        # assert self.n_init_samples <= self.T_max
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
            if self.kwargs["f_type"] in ["HPO", "condHPO"]:
                self.cardinality_xi_t = self.kwargs["cardinality_domain"]
                self.xi_t = None
                self.HPO_X_mean = np.array(self.kwargs["f_opts"]["X_mean"])
                self.HPO_X_std = np.array(self.kwargs["f_opts"]["X_std"])
                self.HPO_X_std = np.where(self.HPO_X_std == 0., 1., self.HPO_X_std)
            else:
                self.cardinality_xi_t = self.kwargs["cardinality_domain"]
                self.xi_t = sobol_seq.i4_sobol_generate(self.D, self.kwargs["cardinality_domain"])
                self.multistart_grid = copy.deepcopy(self.xi_t)

        # the features
        self.features = self.kwargs["features"]
        self.feature_order_eval_envs = ["incumbent", "timestep_perc"]

        # observation space
        self.n_features = D
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

        self.observation_space = gym.spaces.Box(low=-100000.0, high=100000.0,
                                                shape=(self.cardinality_xi_t, self.n_features),
                                                dtype=np.float32)
        # self.pass_X_to_pi = self.kwargs["pass_X_to_pi"]

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


    def seed(self, seed=None):
        # sets up the environment-internal random number generator and seeds it with seed
        self.rng = np.random.RandomState()
        self.seeded_with = seed
        self.rng.seed(self.seeded_with)
        # print('[NAPEnv/seed]', self.seeded_with)

        if hasattr(self, "dataset_counter"):
            delattr(self, "dataset_counter")

    def set_af_functions(self, af_fun):
        # connect the policy with the environment for setting up the adaptive grid

        # if not self.pass_X_to_pi:
        self.af = af_fun
        # else:
        # self.af = lambda state: af_fun(self.X)

    def reset(self):
        if self.reward_transformation == "cumulative":
            self.cumulative_reward = 0
        if self.reward_transformation == "best_regret_timed":
            self.regret_history = []
        if self.do_local_af_opt and not self.f_type in ["HPO"]:
            choice_indices = self.rng.choice(len(self.multistart_grid), self.N_S, replace=False)
            self.xi_init = np.array([self.multistart_grid[i] for i in choice_indices])

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
        self.add_data(x_action, action=action if self.kwargs.get("use_index_speedup", False) else None)  # do this BEFORE calling get_reward()
        reward = self.get_reward(x_action)
        self.update_gp()  # do this AFTER calling get_reward()
        self.optimize_AF()
        next_state = self.get_state(self.xi_t)

        # early stop while training
        done = self.is_terminal() or ((self.t >= 30 and np.min(self.y_max-self.Y) < self.f_opts["min_regret"])and "T_max" in self.kwargs and self.f_type == "GP")

        info = {}
        if done:
            y_diffs = self.y_max - self.Y
            assert (self.y_max < self.Y).sum() == 0, f'[NAPEnv.step()] y_max={self.y_max} < Y={self.Y.max()}'
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

        # import sys
        # import nap.RL.utils_gp
        # sys.modules['fsaf'] = sys.modules['nap']
        # sys.modules['fsaf.RL.gp_utils'] = sys.modules['nap.RL.utils_gp']

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
            

            ssgp = SparseSpectrumGP(seed=seed,
                                    input_dim=self.D,
                                    noise_var=noise_var,
                                    length_scale=lengthscale,
                                    signal_var=signal_var,
                                    n_features=n_features,
                                    kernel=kernel,
                                    periods=self.f_opts["periods"])
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

            self.pkl_data = pickle.load(open(self.kwargs["f_opts"]["data"][self.dataset_counter], "rb"))

            if self.kwargs["f_opts"].get("shuffle_and_cutoff", False):
                while len(self.pkl_data["accs"]) < self.kwargs['cardinality_domain']:
                    print(
                        f' sample more datasets -> len({self.kwargs["f_opts"]["data"][self.dataset_counter].split("/")[-1]})={len(self.pkl_data["accs"])}')
                    self.dataset_counter = self.rng.randint(len(self.kwargs["f_opts"]["data"]))
                    additional_data = pickle.load(open(self.kwargs["f_opts"]["data"][self.dataset_counter], "rb"))
                    self.pkl_data["domain"] = np.concatenate((self.pkl_data["domain"], additional_data["domain"]))
                    self.pkl_data["accs"] = np.concatenate((self.pkl_data["accs"], additional_data["accs"]))

                if self.kwargs["f_opts"].get("y_row_sampling", False):
                    yuniq, ycount = np.unique(self.pkl_data["accs"], return_counts=True)
                    counts = {v: c for v, c in zip(yuniq, ycount)}
                    logits = np.array([self.pkl_data["accs"][i] / counts[self.pkl_data["accs"][i]] for i in range(len(self.pkl_data["accs"]))])
                    categorical = torch.distributions.Categorical(logits=torch.from_numpy(logits))
                    selected_rows = categorical.sample((self.kwargs["cardinality_domain"],)).numpy()
                else:
                    selected_rows = self.rng.choice(np.arange(self.pkl_data["domain"].shape[0]),
                                                    self.kwargs["cardinality_domain"])
                self.pkl_data["domain"] = self.pkl_data["domain"][selected_rows]
                self.pkl_data["accs"] = self.pkl_data["accs"][selected_rows]

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

            max_pos, max, _, min = HPO_max_min(data=self.pkl_data)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min

            self.f = lambda x, idx=None: HPO(
                x * self.HPO_X_std[None] + self.HPO_X_mean[None],
                data=self.pkl_data,
                index=idx
            )
        elif self.f_type == "condHPO":

            if not hasattr(self, "dataset_counter"):
                self.dataset_counter = self.rng.randint(len(self.kwargs["f_opts"]["models"]))
            else:
                self.dataset_counter += 1
            if self.dataset_counter >= len(self.kwargs["f_opts"]["models"]):
                self.dataset_counter = 0

            # load model, sample from posterior and use this as dataset
            gp_model_path = self.kwargs["f_opts"]["models"][self.dataset_counter]
            gp = torch.load(gp_model_path)
            likelihood = gp.likelihood
            device = torch.device('cpu')
            gp.to(device)
            likelihood.to(device)

            gp_train_data_X = gp.train_inputs[0]
            gp.eval()
            likelihood.eval()

            if self.f_opts.get("perturb_training_inputs", False):
                # select random subset of training set and perturb it as the GP model is bad far from seen points
                selected_rows = self.rng.choice(np.arange(gp_train_data_X.shape[0]),
                                                    self.kwargs["cardinality_domain"])
                X = gp_train_data_X[selected_rows]

                # perturb continuous dimensions
                contX = X[:, self.f_opts["cont_dims"]]
                num_dims_pert_dist = self.f_opts.get("num_dims_pert_dist", "unif")
                pert_std = self.f_opts.get('num_dims_pert_dist_std', 0.1)
                if num_dims_pert_dist == 'normal':
                    pert = torch.randn(*contX.shape).to(device) * pert_std
                else:
                    pert = torch.rand(*contX.shape).to(device) * pert_std - pert_std / 2
                if hasattr(gp, 'input_transform') and gp.input_transform is not None:
                    # perturb original inputs -> un-transform because calling train_inputs is transformed
                    contX = gp.input_transform._untransform(contX).detach()

                contX = contX + pert

                if self.f_opts.get("normalize_X", False):
                    contXrange = contX.max(0)[0] - contX.min(0)[0]
                    safe_contXrange = torch.where(contXrange == 0., 1., contXrange)
                    contX = (contX - contX.min(0)[0]) / safe_contXrange

                X[:, self.f_opts["cont_dims"]] = contX

                # after perturbing cont_dims, if there are na_dims, we need to correct the ones that are flagged
                # as being imputed. Indeed, some cont_dims had missing values imputed with 0.0 (c.f. HPO-B paper)
                # so we need to set them back to 0.0 if the value in na_dim in 1.0 (flag that the 0.0 value in
                # the cont_dim is a "fake" or imputed 0.0 value)
                if len(self.f_opts["na_dims"]) > 0:
                    for na_dim, related_cont_dim in self.f_opts["na_dims"].items():
                        X[:, related_cont_dim] = torch.where(X[:, na_dim] == 1.0, 0.0, X[:, related_cont_dim])

                # perturb cat dims by find neighbors with max. Hamming distance of 1
                if len(self.f_opts["nom_dims"]) > 0:
                    # Note! only perturb real categorical dimensions and not na-dims, dimensions that are simply a flag
                    # for where NA has been imputed in the cont-dim
                    nomX = X[:, self.f_opts["nom_dims"]]
                    nb_perturbed_pos = self.f_opts.get("nb_perturbed_pos", 1)
                    kid = list(self.f_opts["cat_alphabet"].keys())
                    for r in range(len(selected_rows)):
                        pos = self.rng.choice(len(self.f_opts["nom_dims"]), nb_perturbed_pos)
                        for p in pos:
                            nomX[r, p] = self.rng.choice(self.f_opts["cat_alphabet"][kid[p]], 1).item()
                    X[:, self.f_opts["nom_dims"]] = nomX

                X = X.to(device=device, dtype=torch.float64)
                del selected_rows, pert
            else:
                raise RuntimeError

            with torch.no_grad():
                Y = likelihood(gp(X)).sample()
            # make sure y is in [0,1]
            normY = (Y - Y.min()) / (Y.max() - Y.min())

            if hasattr(gp, 'input_transform') and gp.input_transform is not None:
                X = gp.input_transform(X).detach()

            self.pkl_data = {"accs": normY.cpu().numpy(), "domain": X.cpu().numpy()}
            del gp, X, Y, normY

            self.xi_t = get_HPO_domain(data=self.pkl_data)

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

            max_pos, max, _, min = HPO_max_min(data=self.pkl_data)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min
            self.f = lambda x, idx=None: HPO(
                x * self.HPO_X_std[None] + self.HPO_X_mean[None],
                data=self.pkl_data,
                index=idx
            )

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
            # values are standardized, we normalize them for the Transformer
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

            self.f = lambda x, idx=None: Antigen(
                x,
                data=self.pkl_data,
                index=idx
            )
        elif self.f_type == "condAntigen":

            if not hasattr(self, "dataset_counter"):
                self.dataset_counter = self.rng.randint(len(self.kwargs["f_opts"]["data"]))
            else:
                self.dataset_counter += 1
            if self.dataset_counter >= len(self.kwargs["f_opts"]["data"]):
                self.dataset_counter = 0

            # load model, sample from posterior and use this as dataset
            gp = torch.load(self.kwargs["f_opts"]["models"][self.dataset_counter])
            gp.eval()
            if self.f_opts.get("perturb_training_inputs", False):
                # select random subset of training set and perturb it as the GP model is bad far from seen points
                selected_rows = self.rng.choice(np.arange(gp.model.train_inputs[0].shape[0]),
                                                self.kwargs["cardinality_domain"])
                Xcat = gp.model.train_inputs[0][selected_rows, :][:, self.f_opts["cat_dims"]]
                kid = list(self.f_opts["cat_alphabet"].keys())
                for r in range(len(selected_rows)):
                    pos = self.rng.choice(len(self.f_opts["cat_dims"]), self.f_opts.get("nb_perturbed_pos", 1))
                    for p in pos:
                        # Xcat[r, p] = self.rng.choice(np.arange(self.f_opts["cat_alphabet"][p]), 1).item()
                        Xcat[r, p] = self.rng.choice(self.f_opts["cat_alphabet"][kid[p]], 1).item()
            else:
                raise RuntimeError(f"condAntigen without perturbing GP train_inputs is not implemented yet.")

            # sample from posterior MVN results outside [0,1] and also we trained the GP on stdY, so it's predicting
            # centered values (supposedly)
            with torch.no_grad():
                Y = gp.likelihood(gp(Xcat)).sample()
            # make sure y is in [0,1]
            normY = (Y - Y.min()) / (Y.max() - Y.min())

            self.pkl_data = {'domain': Xcat.cpu().numpy(), 'accs': normY.cpu().numpy()}
            del gp, selected_rows, Xcat, Y, normY

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

            self.f = lambda x, idx=None: Antigen(
                x,
                data=self.pkl_data,
                index=idx
            )

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
        normalizer = False

        # this is only dummy data as GPy is not able to have empty training set
        # for prediction, the GP is not used for empty training set
        if self.n_init_samples > 0:
            if self.kwargs.get('use_index_speedup', False):
                X = self.xi_t[self.initial_design, :].reshape(self.initial_design.size, self.D)
            else:
                X = self.initial_design.reshape(self.n_init_samples, self.D)
            Y = []
            # breakpoint()
            for inid in self.initial_design:
                if self.kwargs.get('use_index_speedup', False):
                    Y.append(self.pkl_data["accs"][inid])
                else:
                    Y.append(self.f(inid.reshape(1,-1)))
            Y = np.array(Y).reshape(self.n_init_samples,1)
            self.X = X
            self.Y = Y

        else:
            self.X = self.Y = None
        self.gp_is_empty = True

    def add_data(self, X, action=None):
        assert X.ndim == 2

        # evaluate f at X and add the result to the GP
        if action is None:
            Y = self.f(X)
        else:
            Y = np.array(self.pkl_data["accs"][action]).reshape(-1, 1)

        if self.X is None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)

        self.t += X.shape[0]

    def update_gp(self):
        assert self.Y is not None
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
        if self.kwargs.get("use_index_speedup", False):
            shuffled_index = np.arange(X.shape[0])
            self.rng.shuffle(shuffled_index)
            X = X[shuffled_index]
            self.xi_t = X
            self.pkl_data['accs'] = self.pkl_data['accs'][shuffled_index]
        else:
            shuffled_index = None
            self.rng.shuffle(X)
        # breakpoint()
        feature_count = 0
        idx = 0
        n_train = 0 if self.X is None else self.X.shape[0]
        state = np.zeros((self.T + self.n_init_samples + X.shape[0], self.n_features - 2), dtype=np.float32)

        Y_train = np.zeros((self.T + self.n_init_samples + X.shape[0], 1), dtype=np.float32)
        if self.Y is not None:
            Y_train[:n_train] = self.Y

        feature_count += self.D
        if self.X is not None:
            state[:n_train] = self.X

        state[self.T + self.n_init_samples:] = X
        idx += self.D

        state_x_indep = np.array([self.get_incumbent(), self.t / self.T])

        if self.kwargs["f_type"] in ["HPO", "condHPO", "Antigen", "condAntigen"]:
            # breakpoint()
            if self.kwargs.get("use_index_speedup", False):
                fX = np.array(self.pkl_data['accs']).reshape(-1, 1)
            else:
                fX = np.array([self.f(X[i]) for i in range(X.shape[0])])[:, 0]
        else:
            fX = self.f(X)

        if self.kwargs.get("online_ystd", False):
            y_range = self.kwargs.get("y_range", None)
            y_range = np.array(y_range)
            if self.Y is None:
                train_mean = 0.
                train_std = 1.
            elif len(self.Y) == 1:
                train_mean = self.Y.mean()
                train_std = 1.
            else:  # len(self.Y) > 1
                train_mean = self.Y.mean()
                train_std = self.Y.std()

            Y_train_std = np.copy(Y_train)
            Y_train_std[:n_train] = (Y_train[:n_train] - train_mean) / train_std
            fX_std = (fX - train_mean) / train_std

            if y_range is not None:
                Y_train_std = np.clip(Y_train_std, -y_range + 1e-3, y_range - 1e-3)
                fX_std = np.clip(fX_std, -y_range + 1e-3, y_range - 1e-3)

            return state, Y_train_std, n_train, state_x_indep, fX_std, self.T + self.n_init_samples, self.n_init_samples

        return state, Y_train, n_train, state_x_indep, fX, self.T + self.n_init_samples, self.n_init_samples

    def get_reward(self, X):
        # make sure you already increased the step counter self.t before calling this method!
        # make sure you updated the training set but did NOT update the gp before calling this method!
        assert self.Y is not None  # this method should not be called with empty training set
        negativity_check = False

        # compute the simple regret
        y_diffs = self.y_max - self.Y
        assert (self.y_max < self.Y).sum() == 0, f'[NAPEnv.get_reward()] y_max={self.y_max} < Y={self.Y.max()}'
        simple_regret = np.min(y_diffs)
        reward = np.asscalar(simple_regret)

        # apply reward transformation
        if self.reward_transformation == "none":
            reward = reward
        elif self.reward_transformation == "cumulative":
            # breakpoint()
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
        elif self.reward_transformation == "best_regret_timed":
            self.regret_history.append(reward)
            if self.is_terminal():
                self.regret_history = np.array(self.regret_history)
                ind = np.argmin(self.regret_history)
                min_regret = self.regret_history[ind]
                if min_regret < 0.000001:
                    reward = (self.T - ind) / self.T
                else:
                    reward = -min_regret
            else:
                reward = 0.0
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
            # breakpoint()
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
