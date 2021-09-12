""" Contains code for the shapes model """

import itertools
from typing import Union, Optional

import numpy as np
import torch
from torch import nn, distributions, Tensor
from torchvision.utils import make_grid

# My imports
from weighted_retraining.weighted_retraining.models import BaseVAE, UnFlatten, MLPRegressor


class TopologyMLPRegressor(MLPRegressor):
    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        super().__init__(input_dim, output_dim, *h_dims)

    def forward(self, z: Tensor):
        h = super().forward(z)
        # Activation function should be chosen w.r.t. the expected range of outputs (topology: positive values)
        return torch.relu(h)


def _build_encoder(latent_dim: int):
    model = nn.Sequential(
        # Many convolutions
        nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        ),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        ),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        ),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        ),
        nn.LeakyReLU(),
        # Flatten and FC layers
        nn.Flatten(),
        nn.Linear(in_features=3200, out_features=800),
        nn.LeakyReLU(),
        nn.Linear(in_features=800, out_features=2 * latent_dim),
    )
    return model


def _build_encoder_paper(latent_dim: int):
    model = nn.Sequential(
        # Many convolutions
        nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2, padding=0
        ),
        nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        ),
        nn.ReLU(),
        nn.Dropout2d(
            p=0.2
        ),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2, padding=0
        ),
        nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        # Flatten and FC layers
        nn.Flatten(),
        nn.Linear(in_features=576, out_features=144),
        nn.ReLU(),
        nn.Linear(in_features=144, out_features=2 * latent_dim),
    )
    return model


def _build_decoder(latent_dim: int):
    model = nn.Sequential(
        # FC layers
        nn.Linear(in_features=latent_dim, out_features=800),
        nn.LeakyReLU(),
        nn.Linear(in_features=800, out_features=3200),
        nn.LeakyReLU(),
        # Unflatten
        UnFlatten(128, 5),
        nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.LeakyReLU(),
        # nn.Sigmoid()
    )
    return model


def _build_decoder_paper(latent_dim: int):
    model = nn.Sequential(
        # FC layers
        nn.Linear(in_features=latent_dim, out_features=144),
        nn.ReLU(),
        nn.Linear(in_features=144, out_features=576),
        nn.ReLU(),
        # Unflatten
        UnFlatten(64, 3),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.Dropout2d(
            p=0.2
        ),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0,
        ),
        nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
        ),
        nn.Sigmoid()
    )
    return model


class TopologyVAE(BaseVAE):
    """ Convolutional VAE for encoding/decoding 64x64 images """

    def __init__(self, hparams):
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = _build_encoder(self.latent_dim)

        self.decoder = _build_decoder(self.latent_dim)

        self.target_predictor: Optional[TopologyMLPRegressor] = None
        self.pred_loss = nn.MSELoss()
        if self.predict_target:
            self.target_predictor = TopologyMLPRegressor(hparams.latent_dim, 1, *hparams.target_predictor_hdims)

        if hasattr(hparams, "use_binary_data"):
            self.use_binary_data = hparams.use_binary_data
        else:
            self.use_binary_data = False

    def build_target_predictor(self):
        self.target_predictor = TopologyMLPRegressor(self.latent_dim, 1, *self.target_predictor_hdims)

    def encode_to_params(self, x: Tensor):
        if x.ndim == 3:
            x.unsqueeze(1)
        enc_output = self.encoder(x)
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim:]
        return mu, logstd

    def _decoder_loss_bernoulli(self, z, x_orig, return_batch: Optional[bool] = False):
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        dist = distributions.Bernoulli(logits=logits)
        # dist = distributions.ContinuousBernoulli(logits=logits)
        # return -dist.log_prob(x_orig).sum() / z.shape[0]
        if x_orig.ndim < logits.ndim:
            x_orig = x_orig.unsqueeze(1)
        if return_batch:
            return -dist.log_prob(x_orig)
        else:
            return -dist.log_prob(x_orig).sum() / z.shape[0]

    def _decoder_loss_bce(self, z, x_orig, return_batch: Optional[bool] = False):
        """ return binary cross entropy """
        logits = self.decoder(z)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(40, 40).to(z), reduction='none')
        if x_orig.ndim < logits.ndim:
            x_orig = x_orig.unsqueeze(1)
        losses = criterion(logits, x_orig)
        if return_batch:
            return losses
        else:
            return losses.sum() / z.shape[0]

    def _decoder_loss_cos(self, z, x_orig, return_batch: Optional[bool] = False):
        """ return 1-cosine similarity """
        x_recon = torch.sigmoid(self.decoder(z))
        if x_orig.ndim < x_recon.ndim:
            x_orig = x_orig.unsqueeze(1)
        assert x_orig.shape == x_recon.shape, f"{x_orig.shape} and {x_recon.shape} should be the same"
        x_recon_flat = x_recon.view(*x_recon.shape[:-2], -1)
        x_orig_flat = x_orig.view(*x_orig.shape[:-2], -1)
        cos = nn.CosineSimilarity(dim=-1)
        similarity = cos(x_orig_flat, x_recon_flat)
        if return_batch:
            return 1 - similarity
        else:
            return 1 - similarity.mean()

    def _decoder_loss_mse(self, z, x_orig, return_batch: Optional[bool] = False):
        """ return mse_loss """
        x_recon = torch.sigmoid(self.decoder(z))
        if x_orig.ndim < x_recon.ndim:
            x_orig = x_orig.unsqueeze(1)
        assert x_orig.shape == x_recon.shape, f"{x_orig.shape} and {x_recon.shape} should be the same"
        mse = nn.MSELoss(reduction='none')
        losses = mse(x_orig, x_recon)
        if return_batch:
            return losses
        else:
            return losses.mean()

    def decoder_loss(self, z, x_orig, return_batch: Optional[bool] = False, loss: Optional[str] = 'mse'):
        if loss == 'bce':
            return self._decoder_loss_bce(z, x_orig, return_batch)
        elif loss == 'bernoulli':
            return self._decoder_loss_bernoulli(z, x_orig, return_batch)
        elif loss == 'cos':
            return self._decoder_loss_cos(z, x_orig, return_batch)
        elif loss == 'mse':
            return self._decoder_loss_mse(z, x_orig, return_batch)
        else:
            raise ValueError(f"loss method unknown: {loss}")

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return torch.sigmoid(logits)

    def validation_step(self, *args, **kwargs):
        super().validation_step(*args, **kwargs)

        # Visualize latent space
        visualize_latent_space(self, 20)

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


def visualize_latent_space(model: Union[TopologyVAE], nrow: int) -> None:
    # Currently only support 2D manifold visualization
    if model.latent_dim == 2:
        # Create latent manifold
        unit_line = np.linspace(-4, 4, nrow)
        latent_grid = list(itertools.product(unit_line, repeat=2))
        latent_grid = np.array(latent_grid, dtype=np.float32)
        z_manifold = torch.as_tensor(latent_grid, device=model.device)

        # Decode latent manifold
        with torch.no_grad():
            img = model.decode_deterministic(z_manifold).detach().cpu()
        img = torch.clamp(img, 0.0, 1.0)

        # Make grid
        img = make_grid(img, nrow=nrow, padding=5, pad_value=0.5)

        # Log image
        model.logger.experiment.add_image(
            "latent manifold", img, global_step=model.global_step
        )
