""" code for base VAE model """

import argparse
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as torch_func
from torch import nn, Tensor
from torch.nn import functional as F

from utils.utils_cmd import parse_dict, parse_list
from weighted_retraining.weighted_retraining.metrics import ContrastiveLossTorch, METRIC_LOSSES, Required, \
    TripletLossTorch, LogRatioLossTorch


class BaseVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.save_hyperparameters()
        self.latent_dim: int = hparams.latent_dim
        if not hasattr(hparams, 'predict_target'):  # backward compatibility
            hparams.predict_target = False
        self.predict_target: bool = hparams.predict_target

        # Register buffers for prior
        self.register_buffer("prior_mu", torch.zeros([self.latent_dim]))
        self.register_buffer("prior_sigma", torch.ones([self.latent_dim]))

        # Create beta
        self.beta = hparams.beta_final
        self.beta_final = hparams.beta_final
        self.beta_annealing = False
        if hparams.beta_start is not None:
            self.beta_annealing = True
            self.beta = hparams.beta_start
            assert (
                    hparams.beta_step is not None
                    and hparams.beta_step_freq is not None
                    and hparams.beta_warmup is not None
            )

        self.logging_prefix = None
        self.log_progress_bar = False

        if not hasattr(hparams, 'metric_loss'):  # backward compatibility
            hparams.metric_loss = None
        if not hasattr(hparams, 'beta_metric_loss'):
            hparams.beta_metric_loss = 1.
        if not hasattr(hparams, 'beta_target_pred_loss'):
            hparams.beta_target_pred_loss = 1.
        self.metric_loss = hparams.metric_loss
        self.metric_loss_kw = {}
        self.beta_metric_loss = hparams.beta_metric_loss
        self.beta_target_pred_loss = hparams.beta_target_pred_loss
        if self.metric_loss is not None:
            assert self.metric_loss in METRIC_LOSSES
            if self.metric_loss in ('contrastive', 'triplet', 'log_ratio'):
                for kw, default in METRIC_LOSSES[self.metric_loss]['kwargs'].items():
                    if kw in hparams.metric_loss_kw:
                        self.metric_loss_kw[kw] = hparams.metric_loss_kw[kw]
                    elif not isinstance(default, Required):
                        self.metric_loss_kw[kw] = default
                    else:
                        raise ValueError(f'Should specify {kw} in --metric_loss_kw dictionary as it is required from '
                                         f'metric loss {hparams.metric_loss}: {hparams.metric_loss_kw}')

    @property
    def require_ys(self):
        """ Whether (possibly transformed) target values are required in forward method """
        if self.predict_target:
            return True
        if self.metric_loss is not None:
            if self.metric_loss in ('contrastive', 'triplet', 'log_ratio'):
                return True
            else:
                raise ValueError(f'{self.metric_loss} not supported')
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.register('type', list, parse_list)
        parser.register('type', dict, parse_dict)

        vae_group = parser.add_argument_group("VAE")
        vae_group.add_argument("--latent_dim", type=int, required=True, help='Dimensionality of the latent space')
        vae_group.add_argument("--lr", type=float, default=1e-3, help='Learning rate')
        vae_group.add_argument("--beta_final", type=float, default=1.0, help='Final value for beta')
        vae_group.add_argument(
            "--beta_start",
            type=float,
            default=None,
            help="starting beta value; if None then no beta annealing is used",
        )
        vae_group.add_argument(
            "--beta_step",
            type=float,
            default=None,
            help="multiplicative step size for beta, if using beta annealing",
        )
        vae_group.add_argument(
            "--beta_step_freq",
            type=int,
            default=None,
            help="frequency for beta step, if taking a step for beta",
        )
        vae_group.add_argument(
            "--beta_warmup",
            type=int,
            default=None,
            help="number of iterations of warmup before beta starts increasing",
        )
        vae_group.add_argument(
            "--predict_target",
            action='store_true',
            help="Generative model predicts target value",
        )
        vae_group.add_argument(
            "--target_predictor_hdims",
            type=list,
            default=None,
            help="Hidden dimensions of MLP predicting target values",
        )
        vae_group.add_argument(
            "--beta_target_pred_loss",
            type=float,
            default=1.,
            help="Weight of the target_prediction loss added in the ELBO",
        )
        parser.add_argument(
            "--beta_metric_loss",
            type=float,
            default=1.,
            help="Weight of the metric loss added in the ELBO",
        )
        vae_group = parser.add_argument_group("Metric learning")
        vae_group.add_argument(
            "--metric_loss",
            type=str,
            help="Metric loss to add to VAE loss during training of the generative model to get better "
                 "structured latent space (see `METRIC_LOSSES`), "
                 "must be one of ['contrastive', 'triplet', 'log_ratio', 'infob']",
        )
        vae_group.add_argument(
            "--metric_loss_kw",
            type=dict,
            default=None,
            help="Threshold parameter for metric loss, "
                 "must be one of [{'threshold':.1}, {'theshold':.1,'margin':1}, {'threshold':.1,'soft':True}, "
                 "{'threshold':.1,'hard':True}]",
        )
        return parser

    def target_prediction_loss(self, z: Tensor, target: Tensor):
        """ Return MSE loss associated to target prediction

        Args:
            z: latent variable
            target: ground truth score
        """
        y_pred = self.target_predictor(z)
        assert y_pred.shape == target.shape, (y_pred.shape, target.shape)
        pred_loss = self.pred_loss(y_pred, target)
        return pred_loss

    def sample_latent(self, mu, logstd):
        scale_safe = torch.exp(logstd) + 1e-10
        encoder_distribution = torch.distributions.Normal(loc=mu, scale=scale_safe)
        z_sample = encoder_distribution.rsample()
        return z_sample

    def kl_loss(self, mu, logstd, z_sample):
        # Manual formula for kl divergence (more numerically stable!)
        kl_div = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1.0 - 2 * logstd)
        loss = kl_div.sum() / z_sample.shape[0]
        return loss

    def forward(self, *inputs: Tensor, validation: bool = False, m: Optional[float] = None, M: Optional[float] = None):
        """ calculate the VAE ELBO """
        if self.require_ys:
            x, y = inputs[:-1], inputs[-1]
            if len(inputs) == 2:
                x = x[0]
            elif len(inputs) == 1:
                x, y = inputs[0][:-1], inputs[0][-1]
        elif len(inputs) == 1:
            x = inputs[0]
        elif len(inputs) == 2:  # e.g. validation step in semi-supervised setup but we have targets that we do not use
            x, y = inputs[0], inputs[1]
        else:
            x = inputs
        # reparameterization trick
        mu, logstd = self.encode_to_params(x)
        z_sample = self.sample_latent(mu, logstd)

        # KL divergence and reconstruction error
        kl_loss = self.kl_loss(mu, logstd, z_sample)
        reconstruction_loss = self.decoder_loss(z_sample, x)

        # Final loss
        if validation:
            beta = self.beta_final
        else:
            beta = self.beta

        prediction_loss = 0
        if self.predict_target:
            if self.predict_target:
                if y.shape[-1] != 1:
                    y = y.unsqueeze(-1)
            prediction_loss = self.target_prediction_loss(z_sample, target=y)

        metric_loss = 0
        if self.metric_loss is not None:
            if self.predict_target:
                assert m is not None and M is not None and M >= m, (m, M)
                # if target prediction is used, the target values should be normalised at this stage as we need them
                # to be within (0,1) for the metric loss
                if m == M:
                    y[:] = 0
                else:
                    y = (y - m) / (M - m)
            if y.shape[-1] != 1:
                y = y.unsqueeze(-1)
            assert y.min().item() >= -1e-5 and y.max().item() <= (1 + 1e-5), (y.min(), y.max())
            if self.metric_loss == 'contrastive':
                constr_loss = ContrastiveLossTorch(threshold=self.metric_loss_kw['threshold'],
                                                   hard=self.metric_loss_kw.get('hard'))
                metric_loss = constr_loss(z_sample, y)
            elif self.metric_loss == 'triplet':
                triplet_loss = TripletLossTorch(
                    threshold=self.metric_loss_kw['threshold'],
                    margin=self.metric_loss_kw.get('margin'),
                    soft=self.metric_loss_kw.get('soft'),
                    eta=self.metric_loss_kw.get('eta')
                )
                metric_loss = triplet_loss(z_sample, y)
            elif self.metric_loss == 'log_ratio':
                log_ratio_loss = LogRatioLossTorch()
                metric_loss = log_ratio_loss(z_sample, y)
            else:
                raise ValueError(f'{self.metric_loss} not supported')

        loss = reconstruction_loss \
               + beta * kl_loss \
               + self.beta_target_pred_loss * prediction_loss \
               + self.beta_metric_loss * metric_loss

        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"kl/{self.logging_prefix}", kl_loss, prog_bar=self.log_progress_bar
            )
            if self.predict_target:
                self.log(
                    f"pred_target/{self.logging_prefix}", prediction_loss, prog_bar=self.log_progress_bar
                )
            if self.metric_loss is not None:
                self.log(
                    f"metric_loss:{self.metric_loss}/{self.logging_prefix}", metric_loss, prog_bar=self.log_progress_bar
                )
            self.log(f"loss/{self.logging_prefix}", loss)
        return loss

    def sample_prior(self, n_samples):
        return torch.distributions.Normal(self.prior_mu, self.prior_sigma).sample(
            torch.Size([n_samples])
        )

    def _increment_beta(self):

        if not self.beta_annealing:
            return

        # Check if the warmup is over and if it's the right step to increment beta
        if (
                self.global_step > self.hparams.beta_warmup
                and self.global_step % self.hparams.beta_step_freq == 0
        ):
            # Multiply beta to get beta proposal
            self.beta = min(self.hparams.beta_final, self.beta * self.hparams.beta_step)

    # Methods to overwrite (ones that differ between specific VAE implementations)
    def encode_to_params(self, x):
        """ encode a batch to it's distributional parameters """
        raise NotImplementedError

    def decoder_loss(self, z: torch.Tensor, x_orig) -> torch.Tensor:
        """ Get the loss of the decoder given a batch of z values to decode """
        raise NotImplementedError

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx, m: Optional[float] = None, M: Optional[float] = None):
        if hasattr(self.hparams, 'cuda') and self.hparams.cuda is not None:
            self.log(f"cuda:{self.hparams.cuda}",
                     pl.core.memory.get_gpu_memory_map()[f'gpu_id: {self.hparams.cuda}/memory.used (MB)'],
                     prog_bar=True)
        self._increment_beta()
        self.log("beta", self.beta, prog_bar=True)

        self.logging_prefix = "train"
        loss = self(*batch, m=m, M=M)
        self.logging_prefix = None
        return loss

    def validation_step(self, batch, batch_idx, m: Optional[float] = None, M: Optional[float] = None):
        if hasattr(self.hparams, 'cuda') and self.hparams.cuda is not None:
            self.log(f"cuda:{self.hparams.cuda}",
                     pl.core.memory.get_gpu_memory_map()[f'gpu_id: {self.hparams.cuda}/memory.used (MB)'],
                     prog_bar=True)
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(*batch, validation=True, m=m, M=M)
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # No scheduling
        sched = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=.2, patience=1,
                                                                         min_lr=self.hparams.lr),
                 'interval': 'epoch',
                 'monitor': 'loss/val'
                 }
        return dict(optimizer=opt,
                    lr_scheduler=sched)


class Projection(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class BaseCLR(BaseVAE):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.temperature = 0.1
        self.projection = Projection(input_dim=hparams.latent_dim, hidden_dim=32, output_dim=32)

    def forward(self, *inputs: Tensor, validation: bool = False):
        """ calculate the VAE ELBO """
        x, x2 = inputs
        assert x.shape == x2.shape
        # Reparameterization trick input 1
        mu, logstd = self.encode_to_params(x)
        z_sample = self.sample_latent(mu, logstd)

        # Reparameterization trick input 2
        mu2, logstd2 = self.encode_to_params(x2)
        z_sample2 = self.sample_latent(mu2, logstd2)

        # get z representations
        h1 = self.projection(mu)
        h2 = self.projection(mu2)

        # Contrastive Latent Loss
        contrastive_latent_loss = self.nt_xent_loss(h1, h2, self.temperature)

        # KL divergence and reconstruction error averaged over both inputs
        kl_loss = self.kl_loss(mu, logstd, z_sample) / 2.0
        reconstruction_loss = self.decoder_loss(z_sample, x) / 2.0

        kl_loss += self.kl_loss(mu2, logstd2, z_sample2) / 2.0
        reconstruction_loss += self.decoder_loss(z_sample2, x2) / 2.0

        # Final loss
        if validation:
            beta = self.beta_final
        else:
            beta = self.beta

        loss = reconstruction_loss + beta * kl_loss + contrastive_latent_loss
        # loss = reconstruction_loss + beta * kl_loss

        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"kl/{self.logging_prefix}", kl_loss, prog_bar=self.log_progress_bar
            )

            self.log(
                f"contrastive_latent_loss/{self.logging_prefix}", contrastive_latent_loss,
                prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss, prog_bar=self.log_progress_bar)
        return loss

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


class UnFlatten(torch.nn.Module):
    """ unflattening layer """

    def __init__(self, filters=1, size=28):
        super().__init__()
        self.filters = filters
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.filters, self.size, self.size)


class MLPRegressor(torch.nn.Module):
    """ Simple class for regression """

    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        """

        Args:
            input_dim: input dimension
            output_dim: output dimension
            *h_dims: dimensions of the MLP hidden layers
        """
        super(MLPRegressor, self).__init__()
        self.h_dims = list(h_dims)
        layer_dims = [input_dim] + self.h_dims + [output_dim]
        self.layers = torch.nn.ModuleList([nn.Linear(u, v) for u, v in zip(layer_dims[:-1], layer_dims[1:])])

    def forward(self, x: Tensor):
        h = x
        for layer in self.layers[:-1]:
            h = torch_func.relu(layer(h))
        return self.layers[-1](h)
