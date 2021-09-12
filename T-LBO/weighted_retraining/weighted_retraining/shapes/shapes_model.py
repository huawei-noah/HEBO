""" Contains code for the shapes model """

import itertools
from typing import Union, Optional

import numpy as np
import torch
from torch import nn, distributions, Tensor
from torchvision.utils import make_grid

# My imports
from weighted_retraining.weighted_retraining.models import BaseCLR, BaseVAE, UnFlatten, MLPRegressor


class ShapesMLPRegressor(MLPRegressor):
    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        super().__init__(input_dim, output_dim, *h_dims)

    def forward(self, z: Tensor):
        h = super().forward(z)
        # Activation function should be chosen w.r.t. the expected range of outputs (shapes: positive values)
        return torch.relu(h)


def _build_encoder(latent_dim: int):
    model = nn.Sequential(
        # Many convolutions
        nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        # Flatten and FC layers
        nn.Flatten(),
        nn.Linear(in_features=256, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=2 * latent_dim),
    )
    return model


def _build_decoder(latent_dim: int):
    model = nn.Sequential(
        # FC layers
        nn.Linear(in_features=latent_dim, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=256),
        nn.ReLU(),
        # Unflatten
        UnFlatten(16, 4),
        # Conv transpose layers
        nn.ConvTranspose2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            padding=2,
            stride=2,
            output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            padding=2,
            stride=2,
            output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            padding=2,
            stride=2,
            output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=8,
            out_channels=8,
            kernel_size=5,
            padding=2,
            stride=2,
            output_padding=1,
        ),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1,
            output_padding=0,
        ),
    )
    return model


class ShapesVAECLR(BaseCLR):
    """ Convolutional VAE for encoding/decoding 64x64 images """

    def __init__(self, hparams):
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = _build_encoder(self.latent_dim)

        self.decoder = _build_decoder(self.latent_dim)

        self.target_predictor: Optional[ShapesMLPRegressor] = None
        self.pred_loss = nn.MSELoss()
        if self.predict_target:
            self.target_predictor = ShapesMLPRegressor(hparams.latent_dim, 1, *hparams.target_predictor_hdims)

    def encode_to_params(self, x: Tensor):
        enc_output = self.encoder(x)
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim:]
        return mu, logstd

    def decoder_loss(self, z, x_orig):
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        dist = distributions.Bernoulli(logits=logits)
        return -dist.log_prob(x_orig).sum() / z.shape[0]

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


class ShapesVAE(BaseVAE):
    """ Convolutional VAE for encoding/decoding 64x64 images """

    def __init__(self, hparams):
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = _build_encoder(self.latent_dim)

        self.decoder = _build_decoder(self.latent_dim)

        self.target_predictor: Optional[ShapesMLPRegressor] = None
        self.pred_loss = nn.MSELoss()
        if self.predict_target:
            self.target_predictor = ShapesMLPRegressor(hparams.latent_dim, 1, *hparams.target_predictor_hdims)

    def encode_to_params(self, x: Tensor):
        enc_output = self.encoder(x)
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim:]
        return mu, logstd

    def decoder_loss(self, z, x_orig):
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        dist = distributions.Bernoulli(logits=logits)
        return -dist.log_prob(x_orig).sum() / z.shape[0]

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



def visualize_latent_space(model: Union[ShapesVAE], nrow: int) -> None:
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
