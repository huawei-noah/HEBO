"""
Robust model that is compatible with gpytorch and botorch.
"""
from model_utils import model_common_utils as mcu

import gpytorch as gpyt
import botorch as bot
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import transforms as tf
from typing import Any, Optional, Union
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.utils import gpt_posterior_settings
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor
import warnings
from botorch.posteriors.transformed import TransformedPosterior  # pragma: no cover


# %%
class RobustGP(gpyt.models.ExactGP, GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood, num_inputs,
                 input_transform=None, outcome_transform=None, additional_transform=None,
                 **kwargs):
        # Note:
        # we first need to warm up the input and outcome transformers (say the params in
        # the outcome standardize), then save the raw inputs and transformed outcome in the model
        # via feeding them to the ExactGP.__init__().
        # During the training, the input transform is applied before each forward() to
        # normalize the train inputs.
        # After that, if the model.eval() is called, the model will apply the input
        # transformation on its saved train inputs and prepare to concatenate with the transformed
        # test inputs.
        # Moreover, the outcome is untransformed in the posterior() and the additional transform is
        # only applied in the posterior
        """
        A GP model that is compatible with BoTorch
        :param train_inputs: a tensor or a tuple of tensors for training inputs
        :param train_targets: a tensor of training targets
        :param likelihood: likelihood to use
        :param num_inputs: the required number of train inputs
        :param input_transform: transformations to be applied on the inputs, e.g., Normalization
        :param outcome_transform: transformation for the outcome, say standardization
        :param additional_transform: additional transformation to apply on the inputs
        :param kwargs: other model configurations.
        """
        _in_tf = input_transform if input_transform is not None \
            else self.define_default_input_transform(**kwargs)
        _out_tf = outcome_transform if outcome_transform is not None \
            else self.define_default_outcome_transform(**kwargs)

        # Apply the transformers to warm up the params (say the params in standardization)
        with torch.no_grad():
            _in_tf.transform(train_inputs)
        if _out_tf is not None:
            # transform the outcome
            train_targets, _ = _out_tf(train_targets)
            train_targets = train_targets.squeeze(-1)

        # save the raw inputs and transformed outcome
        gpyt.models.ExactGP.__init__(self, train_inputs, train_targets, likelihood)

        self.input_transform = _in_tf
        self.outcome_transform = _out_tf
        self.num_inputs = num_inputs
        self.additional_transform = additional_transform if additional_transform is not None \
            else self.define_additional_transform(**kwargs)

        self.mean_module = self.define_mean_module(**kwargs)
        self.covar_module = self.define_covar_module(**kwargs)

        self.kwargs = kwargs

    def define_default_input_transform(self, **kwargs):
        n_var = kwargs['n_var']
        return tf.Normalize(d=n_var, transform_on_train=True)

    def define_default_outcome_transform(self, m=1, **kwargs):
        return tf.outcome.Standardize(m=m)

    def define_mean_module(self, **kwargs):
        return gpyt.means.ConstantMean()

    def define_covar_module(self, **kwargs):
        return gpyt.kernels.ScaleKernel(gpyt.kernels.MaternKernel())

    def define_additional_transform(self, **kwargs):
        return None

    def transform_inputs_additional(self, X):
        # apply the additional transform
        n_inputs = 1 if isinstance(X, torch.Tensor) else len(X)
        if n_inputs != self.num_inputs:
            if self.additional_transform is not None:
                X = self.additional_transform(X)
            else:
                raise ValueError(f"Expect {self.num_inputs} inputs but found {len(X)} "
                                 f"and no additional transformer.")
        return X

    def forward(self, X):
        # note that we assume X is already applied with additional transform
        if self.training:
            X = self.transform_inputs(X)
        xc_raw = X
        mean_x = self.mean_module(xc_raw)
        covar_x = self.covar_module(xc_raw)
        return gpyt.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X,
            observation_noise: Union[bool, Tensor] = False,
            posterior_transform: Optional[PosteriorTransform] = None,
            **kwargs: Any,
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        self.eval()  # make sure model is in eval mode
        # apply the additional transform
        X = self.transform_inputs_additional(X)

        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            mvn = self(*X) if self.num_inputs > 1 else self(X)  # support multiple inputs
            if observation_noise is not False:
                if isinstance(observation_noise, torch.Tensor):
                    # TODO: Make sure observation noise is transformed correctly
                    self._validate_tensor_args(X=X, Y=observation_noise)
                    if observation_noise.size(-1) == 1:
                        observation_noise = observation_noise.squeeze(-1)
                    mvn = self.likelihood(mvn, X, noise=observation_noise)
                else:
                    mvn = self.likelihood(mvn, X)
        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0] if self.num_inputs == 1 \
                    else self.train_inputs  # support multiple inputs
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0] if self.num_inputs == 1 else self.train_inputs
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )


class RobustGPModel():
    def __init__(self, m_cls, num_inputs=1, **kwargs):
        """
        A holistic model for easy use
        :param m_cls: model class
        :param num_inputs: number of inputs
        :param kwargs: model configurations
        """
        self.model = None
        self.likelihood = None
        self.mll = None
        self.optimizer = None
        self.m_cls = m_cls
        self.num_inputs = num_inputs

        # model config
        self.noise_free = kwargs.get('noise_free', False)
        self.dtype = kwargs.get('dtype', torch.float)
        self.device = kwargs.get('device', torch.device('cpu'))
        self.kwargs = kwargs

    def define_optimizer(self, model: torch.nn.Module, **kwargs):
        optimizer = None
        if not kwargs.get('fit_with_scipy', False):
            lr = kwargs.get('lr', 1e-2)
            optimizer = torch.optim.Adam(
                params=[{'params': model.parameters()}],
                lr=lr
            )

        return optimizer

    def define_likelihood(self, **kwargs):
        noise_free = kwargs.get("noise_free", False)
        if noise_free:
            lkh = gpyt.likelihoods.GaussianLikelihood()
            lkh.noise = mcu.NOISE_LB
            lkh.raw_noise.requires_grad = False
        else:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constr = kwargs.get("noise_constr", None)
            lkh = gpyt.likelihoods.GaussianLikelihood(
                noise_prior=noise_prior, noise_constraint=noise_constr
            )

        return lkh

    def define_model(self, tr_x, tr_y, likelihood, **kwargs):
        model = self.m_cls(tr_x, tr_y, likelihood, self.num_inputs, **kwargs)
        return model

    def define_mll(self, likelihood, model):
        return gpyt.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def post_initialize(self, tr_x, tr_y):
        self.likelihood = self.define_likelihood(**self.kwargs)
        self.model = self.define_model(tr_x, tr_y, self.likelihood, **self.kwargs)
        self.mll = self.define_mll(self.likelihood, self.model)

        # dtype and device
        self.likelihood = self.likelihood.to(self.dtype).to(self.device)
        self.model = self.model.to(self.dtype).to(self.device)

    def fit(self, **kwargs):
        """
        Fit the model, retry if NotPSDError happens
        :param tr_x: training inputs
        :param tr_y: training target
        :param kwargs: fit configurations
        :return:
        """
        assert (self.model is not None and self.mll is not None and self.likelihood is not None)
        tr_hist = None
        success = False
        max_retries = kwargs.get('max_retries', 5)
        n_retry = 0
        while not success:
            try:
                with bot.settings.debug(True):
                    tr_hist = self.do_fit(**kwargs)
                success = True
            except Exception as e:
                if n_retry < max_retries:
                    n_retry += 1
                    print(f"[Warn] Model fit fails, retry cnt={n_retry}.", e)
                    success = False
                else:
                    raise e
        return tr_hist

    def do_fit(self, **kwargs):
        self.model.train()
        fit_with_scipy = kwargs.get('fit_with_scipy', True)
        tr_history = None
        if fit_with_scipy:
            bot.fit_gpytorch_mll(self.mll)
        else:
            tr_history = []
            epoch_num = kwargs.get('epoch_num', 100)
            verbose = kwargs.get('verbose', True)
            print_every = kwargs.get('print_every', 10)
            if self.optimizer is None:
                self.optimizer = self.define_optimizer(self.model, **kwargs)
            for ep_i in range(epoch_num):
                def closure():
                    self.optimizer.zero_grad()
                    output = self.model(self.model.train_inputs[0]) if self.num_inputs == 1 \
                        else self.model(*self.model.train_inputs)
                    loss = -self.mll(output, self.model.train_targets)
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
                xc_ls, xc_ls_str = mcu.get_kernel_lengthscale(self.model.covar_module)
                xc_os, xc_os_str = mcu.get_kernel_output_scale(self.model.covar_module)
                y_noise = self.model.likelihood.noise.item()
                tr_history.append((ep_i, loss.item(), xc_ls, xc_os, y_noise))
                # print
                if verbose and ((ep_i % print_every == 0) or (ep_i == epoch_num - 1)):
                    print(f"[epoch{ep_i}] loss={loss.item():.3f}, "
                          f"xc_lscale={xc_ls_str}, "
                          f"xc_oscale={xc_os_str}, "
                          f"y_noise={y_noise:.3f}")
        return tr_history

    def predict(self, X):
        pred = self.model.posterior(X)
        return pred.mean, pred.variance

    def get_posterior(self, X):
        return self.model.posterior(X)
