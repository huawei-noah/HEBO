import warnings
from typing import Optional, Tuple

import botorch
import gpytorch
import torch.nn.functional
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from bo.kernels import *


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kern, outputscale_constraint=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kern is None:
            kern = gpytorch.kernels.RBFKernel()
        self.covar_module = self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# GP Model
def run_test():
    import torch
    import gpytorch
    import pyro
    from pyro.infer.mcmc import NUTS, MCMC
    num_samples = 2
    warmup_steps = 2
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.randint(0, 20, (5, 11))
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.randn((5))
    # We will use the simplest form of GP model, exact inference

    from gpytorch.priors import UniformPrior
    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)

    model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
        return y

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)
    mcmc_run.run(train_x, train_y)
    model.pyro_load_from_samples(mcmc_run.get_samples())
    model.eval()
    test_x = torch.randint(0, 20, (100, 11))
    expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
    output = model(expanded_test_x)
    preds = likelihood(output)
    print(preds.stddev)


class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 outputscale_constraint=None, ard_dims=None, kern=None, MeanMod=ConstantMean, cat_dims=None,
                 batch_shape=torch.Size()):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.mean_module = MeanMod(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint, batch_shape=batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def dmu_dphi(self, num_cats: int, xs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prediction_strategy is None:
            warnings.warn("Warning: model was not in eval mode. It is now.")
            self.eval()
            self(self.train_inputs[0])
        if xs is None:
            xs = self.train_inputs[0]

        # convert to features --> one-hot encoding
        one_hot_xs: torch.Tensor = torch.nn.functional.one_hot(xs.to(torch.int64), num_classes=num_cats).float()
        one_hot_xs.requires_grad_()

        one_hot_xtrain: torch.Tensor = torch.nn.functional.one_hot(self.train_inputs[0].to(torch.int64),
                                                                   num_classes=num_cats).float()

        # K^-1 y_train
        alpha = self.prediction_strategy.lik_train_train_covar.inv_matmul(self.train_targets.unsqueeze(-1))
        dmu_dphi = []

        for one_hot_x in one_hot_xs:
            # compute jacobian of K(xs, x_train)
            outputscales: torch.Tensor = self.covar_module.outputscale
            outputscales = outputscales.flatten()
            dk_dphi = torch.autograd.functional.jacobian(
                func=lambda x: self.covar_module.base_kernel.forward_one_hot(x.unsqueeze(0), one_hot_xtrain).mul(
                    outputscales).squeeze(0),
                inputs=one_hot_x)

            dk_dphi = torch.permute(dk_dphi, dims=(*np.arange(1, dk_dphi.ndim), 0))

            assert dk_dphi.shape == (*xs.shape[1:], num_cats, len(one_hot_xtrain)), (
                dk_dphi.shape, (*xs.shape[1:], num_cats, len(one_hot_xtrain)))
            dmu_dphi.append((dk_dphi @ alpha).detach().squeeze(-1))
        dmu_dphi = torch.stack(dmu_dphi)
        assert dmu_dphi.shape == (*xs.shape, num_cats), (dmu_dphi.shape, (*xs.shape, num_cats))
        return dmu_dphi

    def ag_ev_phi(self, num_cats: int, dmu_dphi: torch.Tensor = None, xs: torch.Tensor = None,
                  n_samples_threshold: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        num_cats: number of categories
        dmu_dphi: matrix of partial derivatives d mu / d phi of shape (n_points, n_dim, n_categories) --> compute it if None
        xs: points for which derivatives have been computed --> assume it is the training points of the GP if None
        n_samples_threshold: if number of samples having feature phi_ij is less than this threshold, AG_ij will be nan

        Returns
        -------
        ag_phi: matrix of averaged gradient, shape (n_dim, n_categories). Can contain nan (see `n_samples_threshold`)
        ev_phi: matrix of empirical variances, shape (n_dim, n_categories). Can contain nan (see `n_samples_threshold`)
        """

        if xs is None:
            xs = self.train_inputs[0]
        if dmu_dphi is None:
            dmu_dphi = self.dmu_dphi(num_cats=num_cats, xs=xs)

        # Here the one-hot encodings correspond to the phi
        one_hot_xs: torch.Tensor = torch.nn.functional.one_hot(xs.to(torch.int64), num_classes=num_cats)

        # average across samples (filtering samples containing the feature)
        ag_phi: torch.Tensor = (dmu_dphi * one_hot_xs).sum(0) / one_hot_xs.sum(0)

        # empirical variance
        ev_phi = (dmu_dphi ** 2 * one_hot_xs).sum(0) / one_hot_xs.sum(0) - ag_phi ** 2

        # set ag to nan for features observed only few times
        ag_phi[one_hot_xs.sum(0) < n_samples_threshold] = np.nan
        ev_phi[one_hot_xs.sum(0) < n_samples_threshold] = np.nan

        assert ag_phi.shape == ev_phi.shape == (*xs.shape[1:], num_cats), (
            ag_phi.shape, ev_phi.shape, (*xs.shape[1:], num_cats))

        return ag_phi, ev_phi


def train_gp(train_x, train_y, use_ard, num_steps, kern='transformed_overlap', hypers={},
             noise_variance=None,
             cat_configs=None,
             search_strategy='local',
             acq='EI',
             num_samples=51,
             warmup_steps=102,
             thinning=1,
             max_tree_depth=6,
             **params):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    （train_x, train_y）: pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    device = train_x.device
    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]

    outputscale_constraint = Interval(0.5, 5.)

    # if search_strategy in ['glocal', 'global', 'batch_local']:
    #     n_constr = GreaterThan(1e-5)
    #     n_prior = LogNormalPrior(-4.63, 0.5)
    #     # Remove constraints for better GP fit
    #     likelihood = GaussianLikelihood(noise_constraint=n_constr, noise_prior=n_prior).to(device=train_x.device, dtype=train_y.dtype)
    # else:
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device,
                                                                          dtype=train_y.dtype)

    ard_dims = train_x.shape[1] if use_ard else None

    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap':
        # if search_strategy in ['glocal', 'global', 'batch_local']:
        #     kernel = TransformedCategorical(ard_num_dims=ard_dims)
        # else:
        kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'ssk':
        kernel = FastStringKernel(seq_length=train_x.shape[1], alphabet_size=params['alphabet_size'],
                                  device=train_x.device)
    elif kern == 'rbfBERT':
        from bo.utils import BERTFeatures, batch_iterator
        bert = BERTFeatures(params['BERT_model'], params['BERT_tokeniser'])
        nm_samples = train_x.shape[0]
        if nm_samples > params['BERT_batchsize']:
            reprsn1 = []
            for x in batch_iterator(train_x, params['BERT_batchsize']):
                features1 = bert.compute_features(x)
                reprsn1.append(features1)
            reprsn1 = torch.cat(reprsn1, 0)
        else:
            reprsn1 = bert.compute_features(train_x)
        reprsn1 = rearrange(reprsn1, 'b l d -> b (l d)')
        if use_pca:
            pca = load(f"{antigen}_pca.joblib")
            scaler = load(f"{antigen}_scaler.joblib")
            reprsn1 = torch.from_numpy(pca.transform(scaler.transform(reprsn1.cpu().numpy())))
        train_x = reprsn1.clone()
        del reprsn1, bert
        ard_dims = train_x.shape[1] if use_ard else None
        kernel = BERTWarpRBF(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    elif kern == 'cosineBERT':
        kernel = BERTWarpCosine(lengthscale_constraint=lengthscale_constraint, ard_num_dims=None)
    elif kern == 'rbf':
        kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    else:
        raise ValueError('Unknown kernel choice %s' % kern)

    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kernel,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if search_strategy in ['glocal', 'global', 'batch_local']:
        if hypers:
            model.load_state_dict(hypers)
        fit_gpytorch_model(mll)
    else:
        # Initialize model hypers
        if hypers:
            model.load_state_dict(hypers)
        else:
            hypers = {}
            hypers["covar_module.outputscale"] = 1.0
            if not isinstance(kernel, FastStringKernel):
                hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
            hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
            model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

        for i in range(num_steps):
            optimizer.zero_grad()
            output = model(train_x, )
            loss = -mll(output, train_y).float()
            loss.backward()
            # print(f"Loss Step {i} = {loss.item()}")
            optimizer.step()

    # Switch to eval mode
    model.eval()
    return model


def load_mcmc_samples_to_model(_model, mcmc_samples) -> None:
    """Load MCMC samples into GPyTorchModel."""
    if "noise" in mcmc_samples:
        _model.likelihood.noise_covar.noise = (
            mcmc_samples["likelihood.noise_prior"]
                .detach()
                .clone()
                .view(_model.likelihood.noise_covar.noise.shape)  # pyre-ignore
                .clamp_min(MIN_INFERRED_NOISE_LEVEL)
        )
    _model.covar_module.base_kernel.lengthscale = (
        mcmc_samples["covar_module.base_kernel.lengthscale_prior"]
            .detach()
            .clone()
            .view(_model.covar_module.base_kernel.lengthscale.shape)  # pyre-ignore
    )
    _model.covar_module.outputscale = (  # pyre-ignore
        mcmc_samples["covar_module.outputscale_prior"]
            .detach()
            .clone()
            .view(_model.covar_module.outputscale.shape)
    )
    _model.mean_module.constant.data = (
        mcmc_samples["mean_module.mean_prior"]
            .detach()
            .clone()
            .view(_model.mean_module.constant.shape)  # pyre-ignore
    )
    if "c0" in mcmc_samples:
        _model.input_transform._set_concentration(  # pyre-ignore
            i=0,
            value=mcmc_samples["c0"]
                .detach()
                .clone()
                .view(_model.input_transform.concentration0.shape),  # pyre-ignore
        )
        _model.input_transform._set_concentration(
            i=1,
            value=mcmc_samples["c1"]
                .detach()
                .clone()
                .view(_model.input_transform.concentration1.shape),  # pyre-ignore
        )
