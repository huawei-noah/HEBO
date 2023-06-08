# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


"""
This FSBO implementation is based on the original implementation from Hadi Samer Jomaa
for his work on "Transfer Learning for Bayesian HPO with End-to-End Landmark Meta-Features"
at the NeurIPS 2021 MetaLearning Workshop

The implementation for Deep Kernel Learning is based on the original Gpytorch example:
https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

"""

import torch
import torch.nn as nn
from gpytorch.kernels import ScaleKernel
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time
import gpytorch
import logging
from scipy.optimize import differential_evolution
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))
from nap.RL.utils_gp import TransformedCategorical, MixtureKernel

np.random.seed(1203)
RandomQueryGenerator = np.random.RandomState(413)
RandomSupportGenerator = np.random.RandomState(413)
RandomTaskGenerator = np.random.RandomState(413)


def EI(incumbent, mu, stddev):
    mu = mu.reshape(-1, )
    stddev = stddev.reshape(-1, )

    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)

    return score


class Metric(object):
    def __init__(self, prefix='train: '):
        self.reset()
        self.message = prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.2f}"

    def update(self, loss, noise, mse):
        self.loss.append(loss.item())
        self.noise.append(noise.item())
        self.mse.append(mse.item())

    def reset(self, ):
        self.loss = []
        self.noise = []
        self.mse = []

    def report(self):
        return self.message.format(loss=np.mean(self.loss),
                                   log_var=np.mean(self.noise),
                                   mse=np.mean(self.mse))

    def get(self):
        return {"loss": np.mean(self.loss),
                "noise": np.mean(self.noise),
                "mse": np.mean(self.mse)}

def totorch(x, device):
    return torch.Tensor(x).to(device)

class DeepKernelGP(nn.Module):
    def __init__(self, input_size, log_dir, seed, hidden_size=[32, 32, 32, 32],
                 max_patience=16, kernel="matern", ard=False, nu=2.5, loss_tol=0.0001,
                 lr=0.001, load_model=False, checkpoint=None, epochs=10000,
                 cat_idx=None, num_classes=None,
                 verbose=False, eval_batch_size=1000):
        super(DeepKernelGP, self).__init__()
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_extractor = MLP(self.input_size, hidden_size=self.hidden_size,
                                     cat_idx=cat_idx, num_classes=num_classes).to(self.device)
        self.kernel_config = {"kernel": kernel, "ard": ard, "nu": nu}
        self.max_patience = max_patience
        self.lr = lr
        self.load_model = load_model
        assert checkpoint != None, "Provide a checkpoint"
        self.checkpoint = checkpoint
        self.epochs = epochs
        self.verbose = verbose
        self.loss_tol = loss_tol
        self.eval_batch_size = eval_batch_size
        self.get_model_likelihood_mll(1)

        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    def get_model_likelihood_mll(self, train_size):

        train_x = torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.kernel_config,
                             dims=self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def train(self):
        if self.load_model:
            assert (self.checkpoint is not None)
            self.load_checkpoint(os.path.join(self.checkpoint, "weights"))

        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        weights = copy.deepcopy(self.state_dict())
        patience = 0
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.lr},
                                      {'params': self.feature_extractor.parameters(), 'lr': self.lr}])

        for _ in range(self.epochs):
            optimizer.zero_grad()
            z = self.feature_extractor(self.X_obs)
            self.model.set_train_data(inputs=z, targets=self.y_obs, strict=False)
            predictions = self.model(z)
            try:
                with torch.set_grad_enabled(True):
                    loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Exception {e}")
                logging.info(f"Exception {e}")
                break

            if self.verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_ + 1, epochs=self.epochs, loss=loss.item(), noise=self.likelihood.noise.item()))
            losses.append(loss.detach().to("cpu").item())
            if best_loss > losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())
            if np.allclose(losses[-1], losses[-2], atol=self.loss_tol):
                patience += 1
            else:
                patience = 0
            if patience > self.max_patience:
                break
        self.load_state_dict(weights)
        logging.info(
            f"Current Iteration: {len(self.y_obs)} | Incumbent {max(self.y_obs)} | Duration {np.round(time.time() - starttime)} | Epochs {_} | Noise {self.likelihood.noise.item()}")
        return losses

    def load_checkpoint(self, checkpoint, print_msg=False):
        if print_msg:
            print("Loading FSBO checkpoint ", checkpoint)
        ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'], strict=True)
        self.likelihood.load_state_dict(ckpt['likelihood'], strict=True)
        self.feature_extractor.load_state_dict(ckpt['net'], strict=True)

    def predict(self, X_pen):
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        z_support = self.feature_extractor(self.X_obs).detach()
        self.model.set_train_data(inputs=z_support, targets=self.y_obs, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(X_pen).detach()
            pred = self.likelihood(self.model(z_query))

        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

        return mu, stddev

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        if X_pen is not None:
            self.X_obs, self.y_obs, X_pen = totorch(X_obs, self.device), totorch(y_obs, self.device).reshape(
                -1), totorch(X_pen, self.device)
            n_samples = len(X_pen)
            scores = []
            self.train()

            for i in range(self.eval_batch_size, n_samples + self.eval_batch_size, self.eval_batch_size):
                temp_X = X_pen[range(i - self.eval_batch_size, min(i, n_samples))]
                mu, stddev = self.predict(temp_X)
                score = EI(max(y_obs), mu, stddev)
                scores += score.tolist()

            scores = np.array(scores)
            candidate = np.argmax(scores)

            return candidate

        else:  # continuous
            self.X_obs, self.y_obs = totorch(X_obs, self.device), totorch(y_obs, self.device).reshape(-1)
            dim = len(X_obs[0])
            best_f = torch.max(self.y_obs).item()

            self.train()
            bounds = tuple([(0, 1) for i in range(dim)])

            def acqf(x):
                # x = np.array(x).reshape(-1,dim)
                with torch.no_grad():
                    x = torch.Tensor(x).reshape(-1, dim).to(self.device)
                    mean, std = self.predict(x)
                ei = EI(best_f, mean, std)
                return -ei

            new_x = self.continuous_maximization(dim, bounds, acqf)

            return new_x

    def continuous_maximization(self, dim, bounds, acqf):
        result = differential_evolution(acqf, bounds=bounds, updating='immediate', workers=1, maxiter=20000,
                                        init="sobol")
        return result.x.reshape(-1, dim)


class FSBO(nn.Module):
    def __init__(self, train_data, valid_data, checkpoint_path, batch_size=64, test_batch_size=64,
                 n_inner_steps=1, kernel="matern", ard=False, nu=2.5, hidden_size=[32, 32, 32, 32],
                 cat_idx=None, num_classes=None, device='cuda'):
        super(FSBO, self).__init__()
        ## GP parameters
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_inner_steps = n_inner_steps
        self.checkpoint_path = checkpoint_path
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        first_dataset = list(self.train_data.keys())[0]
        self.input_size = len(train_data[first_dataset]["X"][0])
        self.hidden_size = hidden_size
        self.feature_extractor = MLP(self.input_size, hidden_size=self.hidden_size,
                                     cat_idx=cat_idx, num_classes=num_classes).to(self.device)
        self.kernel_config = {"kernel": kernel, "ard": ard, "nu": nu}
        self.get_model_likelihood_mll(self.batch_size)
        self.mse = nn.MSELoss()
        self.curr_valid_loss = np.inf
        self.get_tasks()
        self.setup_writers()

        self.train_metrics = Metric()
        self.valid_metrics = Metric(prefix="valid: ")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.checkpoint_path, "log.txt"), level=logging.DEBUG)

        print(self)

    def setup_writers(self, ):
        train_log_dir = os.path.join(self.checkpoint_path, "train")
        os.makedirs(train_log_dir, exist_ok=True)
        self.train_summary_writer = SummaryWriter(train_log_dir)

        valid_log_dir = os.path.join(self.checkpoint_path, "valid")
        os.makedirs(valid_log_dir, exist_ok=True)
        self.valid_summary_writer = SummaryWriter(valid_log_dir)

    def get_tasks(self, ):
        self.tasks = list(self.train_data.keys())
        self.valid_tasks = list(self.valid_data.keys())

    def get_model_likelihood_mll(self, train_size):
        train_x = torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y = torch.ones(train_size).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.kernel_config,
                             dims=self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def epoch_end(self):
        RandomTaskGenerator.shuffle(self.tasks)

    def meta_train(self, epochs=50000, lr=0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)
        for epoch in range(epochs):
            self.train_loop(epoch, optimizer, scheduler)

    def train_loop(self, epoch, optimizer, scheduler=None):
        self.epoch_end()
        assert (self.training)
        for task in self.tasks:
            inputs, labels = self.get_batch(task)
            for _ in range(self.n_inner_steps):
                optimizer.zero_grad()
                z = self.feature_extractor(inputs)
                self.model.set_train_data(inputs=z, targets=labels, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
                mse = self.mse(predictions.mean, labels)
                self.train_metrics.update(loss, self.model.likelihood.noise, mse)
        if scheduler:
            scheduler.step()

        training_results = self.train_metrics.get()
        for k, v in training_results.items():
            self.train_summary_writer.add_scalar(k, v, epoch)
        for task in self.valid_tasks:
            mse, loss = self.test_loop(task, train=False)
            self.valid_metrics.update(loss, np.array(0), mse, )

        logging.info(self.train_metrics.report() + " " + self.valid_metrics.report())
        validation_results = self.valid_metrics.get()
        for k, v in validation_results.items():
            self.valid_summary_writer.add_scalar(k, v, epoch)
        self.feature_extractor.train()
        self.likelihood.train()
        self.model.train()

        if validation_results["loss"] < self.curr_valid_loss:
            self.save_checkpoint(os.path.join(self.checkpoint_path, "weights"))
            self.curr_valid_loss = validation_results["loss"]
        self.valid_metrics.reset()
        self.train_metrics.reset()

    def test_loop(self, task, train):
        (x_support, y_support), (x_query, y_query) = self.get_support_and_queries(task, train)
        z_support = self.feature_extractor(x_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred = self.likelihood(self.model(z_query))
            loss = -self.mll(pred, y_query)
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query)

        return mse, loss

    def get_batch(self, task):
        Lambda, response = np.array(self.train_data[task]["X"]), MinMaxScaler().fit_transform(
            np.array(self.train_data[task]["y"])).reshape(-1, )
        card, dim = Lambda.shape
        support_ids = RandomSupportGenerator.choice(np.arange(card),
                                                    replace=False, size=min(self.batch_size, card))

        inputs, labels = Lambda[support_ids], response[support_ids]
        inputs, labels = totorch(inputs, device=self.device), totorch(labels.reshape(-1, ), device=self.device)
        return inputs, labels

    def get_support_and_queries(self, task, train=False):
        hpo_data = self.valid_data if not train else self.train_data
        Lambda, response = np.array(hpo_data[task]["X"]), MinMaxScaler().fit_transform(
            np.array(hpo_data[task]["y"])).reshape(-1, )
        card, dim = Lambda.shape

        support_ids = RandomSupportGenerator.choice(np.arange(card),
                                                    replace=False, size=min(self.batch_size, card))
        diff_set = np.setdiff1d(np.arange(card), support_ids)
        query_ids = RandomQueryGenerator.choice(diff_set, replace=False, size=min(self.batch_size, len(diff_set)))

        support_x, support_y = Lambda[support_ids], response[support_ids]
        query_x, query_y = Lambda[query_ids], response[query_ids]

        return (totorch(support_x, self.device), totorch(support_y.reshape(-1, ), self.device)), \
               (totorch(query_x, self.device), totorch(query_y.reshape(-1, ), self.device))

    def save_checkpoint(self, checkpoint):
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net': nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, config, dims):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if (config["kernel"] == 'rbf' or config["kernel"] == 'RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None))
        elif (config["kernel"] == 'matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=config["nu"], ard_num_dims=dims if config["ard"] else None))
        elif config["kernel"] == "categorical":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                TransformedCategorical(ard_num_dims=dims if config["ard"] else None))
        elif config["kernel"] == "mixed":
            self.covar_module = ScaleKernel(MixtureKernel(
                continuous_dims=cont_dims,
                categorical_dims=cat_dims,
            ))
        else:
            raise ValueError("[ERROR] the kernel '" + str(
                config["kernel"]) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MLP(nn.Module):
    # TODO add support for categorical input
    def __init__(self, input_size, hidden_size=[32, 32, 32, 32], dropout=0.0, cat_idx=None, num_classes=None):
        super(MLP, self).__init__()
        self.nonlinearity = nn.ReLU()
        self.fc = nn.ModuleList([nn.Linear(in_features=input_size, out_features=hidden_size[0])])
        for d_out in hidden_size[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))
        self.out_features = hidden_size[-1]
        self.dropout = nn.Dropout(dropout)
        self.cat_idx = cat_idx
        self.num_classes = num_classes
        self.d_cat = len(cat_idx) if cat_idx is not None else 0
        self.d_num = input_size - 2 - self.d_cat
        if cat_idx is not None:
            assert self.d_cat == len(self.num_classes)
            if isinstance(self.num_classes, dict):
                self.xcat_encoder = [nn.Linear(len(self.num_classes[k]), input_size) for k in self.num_classes]
            else:
                self.xcat_encoder = [nn.Linear(self.num_classes[c], input_size) for c in range(self.d_cat)]
            self.xcat_encoder = nn.ModuleList(self.xcat_encoder)

    def forward(self, x):
        if self.d_cat is not None and self.d_cat > 0:
            xcat_emb = []
            for i, d in enumerate(self.cat_idx):
                xcat_1hot_d = torch.nn.functional.one_hot(
                    x[..., d].to(int),
                    num_classes=len(self.num_classes[d]) if isinstance(self.num_classes, dict) else self.num_classes[i]
                ).to(x)
                xcat_enc_d = self.xcat_encoder[i](xcat_1hot_d)
                xcat_emb.append(xcat_enc_d)
            xcat_emb = torch.stack(xcat_emb, -2).sum(-2)  # TODO check
            x = xcat_emb

        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        x = self.dropout(x)
        return x
