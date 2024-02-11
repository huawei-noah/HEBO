from model_utils import input_transform as tfx
from model_utils.common_model_parts import MLP, CopyModule
from kernels.mmd_kernel import MMDKernel, additive_RQ_kernel
from models.robust_gp import RobustGP

import gpytorch as gpyt
from botorch.models import transforms as tf
import torch
import warnings
from gpytorch import settings
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal


class MMDGP(RobustGP):
    def __init__(self, train_inputs, train_targets, likelihood, num_inputs,
                 input_transform=None, outcome_transform=None, additional_transform=None,
                 hidden_dims=(4, 2), latent_dim=1, **kwargs):
        super(MMDGP, self).__init__(train_inputs, train_targets, likelihood, num_inputs,
                                    input_transform, outcome_transform,
                                    additional_transform, **kwargs)

        # latent mapping
        self.norm_method = kwargs.get('latent_norm_method', None)
        self.skip_conn = kwargs.get('skip_conn', False)
        if hidden_dims is not None:
            self.latent_dim = latent_dim
            self.latent_mapping_module = MLP(
                train_inputs[0].shape[-1], hidden_dims, latent_dim,
                norm_method=self.norm_method
            )
        else:
            self.latent_dim = train_inputs[0].shape[1] + train_inputs[0].shape[1]
            self.latent_mapping_module = CopyModule()

    def define_default_input_transform(self, **kwargs):
        n_var = kwargs['n_var']
        input_bounds = kwargs['input_bounds']
        return tfx.MultiInputTransform(
            tf1=tf.Normalize(d=n_var, bounds=input_bounds, transform_on_train=True),
            tf2=tf.Normalize(d=n_var, bounds=input_bounds, transform_on_train=True),
            tf3=tfx.DummyTransform(transform_on_train=True),
        )

    def define_covar_module(self, **kwargs):
        xc_kern_params = {}
        xc_lscale_constr = kwargs.get('xc_ls_constr', None)
        if xc_lscale_constr is not None:
            xc_kern_params['lengthscale_constraint'] = xc_lscale_constr

        xc_mmd_inner_k = kwargs.get('base_kernel', None)
        if xc_mmd_inner_k is None:
            xc_mmd_inner_k = additive_RQ_kernel(
                alphas=(0.2, 0.5, 1, 2, 5), ls=1.0, learnable_ls=False
            )

        estimator_name = kwargs.get('estimator_name', 'nystrom')
        chunk_size = kwargs.get('chunk_size', 100)
        sub_samp_size = kwargs.get('sub_samp_size', 100)
        covar_module = gpyt.kernels.ScaleKernel(
            MMDKernel(xc_mmd_inner_k, estimator=estimator_name, sub_samp_size=sub_samp_size,
                      chunk_size=chunk_size)
        )
        return covar_module

    def forward(self, xc_raw, xc_samp, xe):
        # note that we assume X is already applied with additional transform
        # input transform
        X = (xc_raw, xc_samp, xe)
        if self.training:
            X = self.transform_inputs(X)
        mean_x, covar = self.compute_mean_cov(X)
        return gpyt.distributions.MultivariateNormal(mean_x, covar)

    def compute_mean_cov(self, x, **kwargs):
        """
        compute the mean and covariance matrix
        :param x: a tuple of (Xc_raw, Xc_samples, Xe),
        where Xc_raw is raw inputs, size= (M * D),
        Xc_samples represents the nearby samples around the raw inputs, size= M * B * D tensor,
        and Xe is the raw inputs of enumerate features, M * 0 tensor.
        """
        Xc_raw, Xc_samples, Xe = x
        Xe_trans = Xe
        mean_x, covar = None, None
        if Xc_raw.shape[-1] > 0 and Xc_samples.shape[-1] > 0:
            _s = Xc_samples.shape[:-1]
            D = Xc_samples.shape[-1]
            proj_X_samples = self.latent_mapping_module(Xc_samples.view(-1, D)).view(*_s, -1)

            # covar
            with gpyt.settings.debug(True) and gpyt.settings.lazily_evaluate_kernels(False):
                k_c = self.covar_module(proj_X_samples, **kwargs)
            covar = k_c if covar is None else (k_c * covar)

            # mean
            Xc_samples_mean = Xc_samples.mean(dim=-2)
            proj_X_raw = self.latent_mapping_module(Xc_samples_mean)
            mean_x = self.mean_module(proj_X_raw)

        if Xe.shape[-1] > 0:
            Xe_trans = self.xe_transformer(Xe)
            k_e = self.xe_covar_module(Xe_trans, **kwargs)
            covar = k_e if covar is None else (k_e * covar)

        return mean_x, covar

    def define_additional_transform(self, **kwargs):
        xc_sample_size = kwargs.get('xc_sample_size', 1000)
        input_sampling_func = kwargs['input_sampling_func']
        n_var = kwargs['n_var']
        return tfx.AdditionalFeatures(f=tfx.additional_xc_samples, transform_on_train=False,
                                      fkwargs={'n_sample': xc_sample_size, 'n_var': n_var,
                                               'sampling_func': input_sampling_func})

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in
                           zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in
                       zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super(ExactGP, self).__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for i, (train_input, input) in enumerate(zip(train_inputs, inputs)):
                # Make sure the batch shapes agree for training/test data
                # special operations for MMD kernel
                dim_2_concat = -3 if i == 1 else -2
                batch_reserved_dim = -3 if i == 1 else -2
                batch_shape = train_inputs[i].shape[:batch_reserved_dim]
                if batch_shape != train_input.shape[:batch_reserved_dim]:
                    batch_shape = torch.broadcast_shapes(batch_shape,
                                                         train_input.shape[:batch_reserved_dim])
                    train_input = train_input.expand(*batch_shape,
                                                     *train_input.shape[batch_reserved_dim:])
                if batch_shape != input.shape[:batch_reserved_dim]:
                    batch_shape = torch.broadcast_shapes(batch_shape,
                                                         input.shape[:batch_reserved_dim])
                    train_input = train_input.expand(*batch_shape,
                                                     *train_input.shape[batch_reserved_dim:])
                    input = input.expand(*batch_shape, *input.shape[batch_reserved_dim:])
                full_inputs.append(torch.cat([train_input, input], dim=dim_2_concat))

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size(
                [joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(
                    full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)
