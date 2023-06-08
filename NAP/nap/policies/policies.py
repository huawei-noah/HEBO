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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from nap.policies.fsbo import EI as fsbo_ei, DeepKernelGP
from nap.policies.iclr2020_mlp import iclr2020_MLP
from scipy.stats import norm
from scipy.optimize import Bounds
import sobol_seq


class iclr2020_NeuralAF(nn.Module):

    def __init__(self, observation_space, action_space, deterministic, options):
        super(iclr2020_NeuralAF, self).__init__()
        self.N_features = None  # has to be set in init_structure()
        self.deterministic = deterministic

        # initialize the network structure
        self.init_structure(observation_space=observation_space, action_space=action_space, options=options)

        # initialize weights
        self.apply(self.init_weights)

    def init_structure(self, observation_space, action_space, options):
        self.N_features = observation_space.shape[1]

        # activation function
        if options["activations"] == "relu":
            f_act = F.relu
        elif options["activations"] == "tanh":
            f_act = torch.tanh
        else:
            raise NotImplementedError("Unknown activation function!")

        # policy network
        self.N_features_policy = self.N_features
        if "exclude_t_from_policy" in options:
            self.exclude_t_from_policy = options["exclude_t_from_policy"]
            assert "t_idx" in options
            self.t_idx = options["t_idx"]
            self.N_features_policy = self.N_features_policy - 1 if self.exclude_t_from_policy else self.N_features_policy
        else:
            self.exclude_t_from_policy = False
        if "exclude_T_from_policy" in options:
            self.exclude_T_from_policy = options["exclude_T_from_policy"]
            assert "T_idx" in options
            self.T_idx = options["T_idx"]
            self.N_features_policy = self.N_features_policy - 1 if self.exclude_T_from_policy else self.N_features_policy
        else:
            self.exclude_T_from_policy = False

        self.policy_net = iclr2020_MLP(d_in=self.N_features_policy, d_out=1, arch_spec=options["arch_spec"], f_act=f_act)

        # value network
        if "use_value_network" in options and options["use_value_network"]:
            self.use_value_network = True
            self.value_net = iclr2020_MLP(d_in=2, d_out=1, arch_spec=options["arch_spec_value"], f_act=f_act)
            self.t_idx = options["t_idx"]
            self.T_idx = options["T_idx"]
        else:
            self.use_value_network = False

    def forward(self, states):
        assert states.dim() == 3
        assert states.shape[-1] == self.N_features

        # policy network
        mask = [True] * self.N_features
        if self.exclude_t_from_policy:
            mask[self.t_idx] = False
        if self.exclude_T_from_policy:
            mask[self.T_idx] = False
        logits = self.policy_net.forward(states[:, :, mask])
        logits.squeeze_(2)

        # value network
        if self.use_value_network:
            tT = states[:, [0], [self.t_idx, self.T_idx]]
            values = self.value_net.forward(tT)
            values.squeeze_(1)
        else:
            values = torch.zeros(states.shape[0]).to(logits.device)

        return logits, values

    def af(self, state):
        state = torch.from_numpy(state[None, :].astype(np.float32))
        with torch.no_grad():
            out = self.forward(state)
        af = out[0].to("cpu").numpy().squeeze()

        return af

    def act(self, state):
        # here, state is assumed to contain a single state, i.e. no batch dimension
        state = state.unsqueeze(0)  # add batch dimension
        out = self.forward(state)
        logits = out[0]
        value = out[1]
        if self.deterministic:
            action = torch.argmax(logits)
        else:
            distr = Categorical(logits=logits)
            # to sample the action, the policy uses the current PROCESS-local random seed, don't re-seed in pi.act
            action = distr.sample()

        return action.squeeze(0), value.squeeze(0)

    def predict_vals_logps_ents(self, states, actions):
        assert actions.dim() == 1
        assert states.shape[0] == actions.shape[0]
        out = self.forward(states)
        logits = out[0]
        values = out[1]

        distr = Categorical(logits=logits)
        logprobs = distr.log_prob(actions)
        entropies = distr.entropy()

        return values, logprobs, entropies

    def set_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def reset(self):
        pass

    @staticmethod
    def num_flat_features(x):
        return np.prod(x.size()[1:])

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.fill_(0.0)


###############################################################################################################
# https://github.com/boschresearch/NoisyInputEntropySearch/blob/master/core/acquisitions/mes.py
class MES:
    def __init__(self, dim, domain=None, n_samples = 1):
        self.domain = Bounds(np.array([0]*dim), np.array([1]*dim))
        self.gp = None
        self.n_samples = n_samples
        self.y_maxes = None
        self.dim = dim

    def act(self, state, X_target, model_target):
        if not type(np.array([])) == type(state):
            state = state.numpy()
        self.gp = model_target
        self.y_maxes = self._sample_maxes()
        mes = self.af(state, X_target, model_target)
        action = np.argmax(mes.reshape(-1))
        action = torch.tensor([action], dtype=torch.int64)
        return action.squeeze(0), torch.tensor([0])

    def af(self, state, X_target, model_target):
        mean_idx = 0
        mu = state[:, mean_idx].reshape(-1)
        std_idx = 1
        std = state[:, std_idx].reshape(-1)
        if self.y_maxes == None:
            self.gp = model_target
            self.y_maxes = self._sample_maxes()
        gamma_maxes = (self.y_maxes - mu) / std
        mes = 0.5 * gamma_maxes * norm.pdf(gamma_maxes) / (norm.cdf(gamma_maxes)+1e-6) - \
                np.log(norm.cdf(gamma_maxes)+ 1e-6)
        return np.array(mes).reshape(-1)
    def _sample_maxes(self):
        if type(self.gp) == type(None):
            return np.zeros((1))
        dim = self.domain.lb.shape[0]
        x_grid = sobol_seq.i4_sobol_generate(dim, 2000)
        mu, var = self.gp.predict(x_grid)
        std = np.sqrt(var)

        def cdf_approx(z):
            tmp_z = z.reshape(1,-1)
            # ret_val = np.zeros(z.shape)
            ret_val = np.prod(norm.cdf((tmp_z - mu) / std),axis=0).reshape(z.shape)
            return ret_val

        lower = np.max(self.gp.Y)
        upper = np.max(mu + 5*std)
        if cdf_approx(upper) <= 0.75:
            upper += 1

        grid = np.linspace(lower, upper, 100)

        cdf_grid = cdf_approx(grid)
        r1, r2 = 0.25, 0.75

        y1 = grid[np.argmax(cdf_grid >= r1)]
        y2 = grid[np.argmax(cdf_grid >= r2)]

        b = (y1 - y2) / (np.log(-np.log(r2)) - np.log(-np.log(r1)))
        a = y1 + (b * np.log(-np.log(r1)))

        maxes = a - b*np.log(-np.log(np.random.rand(self.n_samples,)))
        return maxes
    def set_requires_grad(self, flag):
        pass

class UCB():
    def __init__(self, feature_order, kappa, D=None, delta=None):
        self.feature_order = feature_order
        self.kappa = kappa
        self.D = D
        self.delta = delta
        assert not (self.kappa == "gp_ucb" and self.D is None)
        assert not (self.kappa == "gp_ucb" and self.delta is None)
        np.random.seed(0)  # make UCB behave deterministically

    def act(self, state):
        if not type(np.array([])) == type(state):
            state = state.numpy()
        ucbs = self.af(state)
        action = np.random.choice(np.flatnonzero(ucbs == ucbs.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        if self.kappa == "gp_ucb":
            # timestep_idx = self.feature_order.index("timestep")
            timestep_idx = self.feature_order.index("timestep_perc")
            timesteps = (state[:, timestep_idx])*30 + 1
        else:
            timesteps = None

        kappa = self.compute_kappa(timesteps)
        ucbs = means + kappa * stds
        return ucbs

    def compute_kappa(self, timesteps):
        # https: // arxiv.org / pdf / 0912.3995.pdf
        # https: // arxiv.org / pdf / 1012.2599.pdf
        if self.kappa == "gp_ucb":
            assert timesteps is not None
            nu = 1
            tau_t = 2 * np.log(timesteps ** (self.D / 2 + 2) * np.pi ** 2 / (3 * self.delta))
            kappa = np.sqrt(nu * tau_t)
        else:
            assert timesteps is None
            kappa = self.kappa
        return kappa

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class EI():
    def __init__(self, feature_order):
        self.feature_order = feature_order

    def act(self, state):
        if not type(np.array([])) == type(state):
            state = state.numpy()
        eis = self.af(state)
        action = np.random.choice(np.flatnonzero(eis == eis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        mask = stds != 0.0
        eis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        zs[mask] = (means[mask] - incumbents[mask]) / stds[mask]
        pdf_zs = norm.pdf(zs)
        cdf_zs = norm.cdf(zs)
        eis[mask] = (means[mask] - incumbents[mask]) * cdf_zs + stds[mask] * pdf_zs
        return eis

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass

class RS():
    def __init__(self):
        pass

    def act(self, state):
        action = np.random.choice(state.shape[0])
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        raise()

    def set_requires_grad(self, flag):
        pass

    def reset(self):
        pass


class PI():
    def __init__(self, feature_order, xi):
        self.feature_order = feature_order
        self.xi = xi

    def act(self, state):
        if not type(np.array([])) == type(state):
            state = state.numpy()
        pis = self.af(state)
        action = np.random.choice(np.flatnonzero(pis == pis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        mask = stds != 0.0
        pis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        zs[mask] = (means[mask] - (incumbents[mask] + self.xi)) / stds[mask]
        cdf_zs = norm.cdf(zs)
        pis[mask] = cdf_zs
        return pis

    def set_requires_grad(self, flag):
        pass

class FSBO_EI(EI):
    def __init__(self, logpath, feature_order, fine_tune, **kwargs):
        super().__init__(feature_order)
        self.dataparallel = False
        self.logpath = logpath
        self.fine_tune = fine_tune
        self.fsbo_model = None
        self.cat_idx = kwargs.get('cat_idx', None)
        self.num_classes = kwargs.get('num_classes', None)

    def act(self, state, y_train, masks_index, state_ix, max_T, mask_shift=None, **kwargs):
        if self.fsbo_model is None:
            # self.logpath = "nap/log/TEST/hpobenchXGB/FSBO-v0/2023-03-22-12-45-01/"
            self.fsbo_model = DeepKernelGP(self.feature_order, self.logpath + "/test_logs", 0, epochs=100,
                                           cat_idx=self.cat_idx, num_classes=self.num_classes,
                                           load_model=True, checkpoint=self.logpath, verbose=False)
            self.fsbo_model.load_checkpoint(self.logpath + "/weights", print_msg=True)

        X_obs = state[:masks_index].to(device=self.fsbo_model.device)
        y_obs = y_train[:masks_index].to(device=self.fsbo_model.device)

        X_query = state[max_T:].to(device=self.fsbo_model.device)
        self.fsbo_model.X_obs = X_obs
        self.fsbo_model.y_obs = y_obs.reshape(-1)

        if self.fine_tune:
            self.fsbo_model.train()

        best_f = torch.max(self.fsbo_model.y_obs).item()
        mean, std = self.fsbo_model.predict(X_query)

        ei = fsbo_ei(best_f, mean, std)

        diff = (state[max_T:][None] == state[:masks_index][:, None])
        diff = diff.all(-1).any(0)
        action = torch.argmax(torch.from_numpy(ei) * (1 - diff.int()))

        return action.squeeze(0), torch.tensor([]), torch.tensor([]), torch.tensor([])

    def state_dict(self):
        return {}

    def to(self, _):
        pass

    def load_state_dict(self, _):
        pass
