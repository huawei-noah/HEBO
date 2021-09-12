""" Contains code for the chem model (JT-VAE) """

import argparse
import torch
from torch import nn, Tensor
import torch.nn.functional as torch_func

# My imports
from typing import List, Any, Optional

from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.models import BaseVAE, MLPRegressor
from weighted_retraining.weighted_retraining.chem.jtnn import JTNNVAE


class BlackBoxMLPRegressor(MLPRegressor):

    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        super().__init__(input_dim, output_dim, *h_dims)

    def forward(self, z: Tensor):
        h = super().forward(z)
        return torch.relu(h)


class JTVAE(BaseVAE):
    """ Junction-tree VAE for chem task, following old style """

    # Default parameters
    hidden_size = 450
    latent_T_size = None
    depthT = 20
    depthG = 3

    def __init__(self, hparams, vocab):
        super().__init__(hparams)

        # Construct jtnn
        self.jtnn_vae = JTNNVAE(
            vocab,
            hparams.hidden_size if hasattr(hparams, 'hidden_size') else self.hidden_size,
            hparams.latent_dim,
            hparams.depthT if hasattr(hparams, 'depthT') else self.depthT,
            hparams.depthG if hasattr(hparams, 'depthG') else self.depthG,
            latent_T_size=hparams.latent_T_size if hasattr(hparams, 'latent_T_size') else self.latent_T_size,
        )

        self.target_predictor: Optional[BlackBoxMLPRegressor] = None
        self.pred_loss = nn.MSELoss()
        if self.predict_target:
            self.target_predictor = BlackBoxMLPRegressor(hparams.latent_dim, 1,
                                                         *hparams.target_predictor_hdims)

        # Init all parameters
        for param in self.jtnn_vae.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        # Turn on flags for backward model compatibility
        self.jtnn_vae.jtnn.GRU.tanh = False
        self.jtnn_vae.decoder.U_i_relu = False
        self.jtnn_vae._no_assm = True

        # target normalisation constants
        self.training_m = None
        self.training_M = None
        self.validation_m = None
        self.validation_M = None

    def build_target_predictor(self):
        self.target_predictor = BlackBoxMLPRegressor(self.latent_dim, 1,
                                                     *self.target_predictor_hdims)

    @staticmethod
    def add_model_specific_args(parent_parser, call_base: bool = True):
        if call_base:
            parent_parser = BaseVAE.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # Model group
        model_group = parser.add_argument_group(title="model")
        model_group.add_argument("--hidden_size", type=int, default=JTVAE.hidden_size)
        model_group.add_argument(
            "--latent_T_size", type=int, default=JTVAE.latent_T_size
        )
        model_group.add_argument("--depthT", type=int, default=JTVAE.depthT)
        model_group.add_argument("--depthG", type=int, default=JTVAE.depthG)

        return parser

    def encode_to_params(self, x):

        # Run encoder
        _, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.jtnn_vae.encode(
            x_jtenc_holder, x_mpn_holder
        )

        # Store the tree mess for use later (admittedly a bit of a hack)
        self._x_tree_mess = x_tree_mess
        self._x_jtmpn_holder = x_jtmpn_holder

        # Run separate NNs to find means/variances
        z_tree_mean = self.jtnn_vae.T_mean(x_tree_vecs)
        z_mol_mean = self.jtnn_vae.G_mean(x_mol_vecs)

        z_tree_logvar = self.jtnn_vae.T_var(x_tree_vecs)
        z_mol_logvar = self.jtnn_vae.G_var(x_mol_vecs)

        # Concatenate variables
        z_mean = torch.cat([z_tree_mean, z_mol_mean], dim=-1)
        z_logvar = torch.cat([z_tree_logvar, z_mol_logvar], dim=-1)

        # Make "old-style" adjustment
        z_logstd = -torch.abs(z_logvar / 2)

        return z_mean, z_logstd

    def decoder_loss(self, z, x_orig, return_batch: bool = False):

        # Process inputs
        x_batch = x_orig[0]
        z_tree = z[:, : self.jtnn_vae.latent_T_size]
        z_mol = z[:, self.jtnn_vae.latent_T_size:]

        # Decoder loss
        word_loss, topo_loss, word_acc, topo_acc = self.jtnn_vae.decoder(
            x_batch, z_tree, return_batch=return_batch
        )
        assm_loss, assm_acc = self.jtnn_vae.assm(
            x_batch, self._x_jtmpn_holder, z_mol, self._x_tree_mess, return_batch=return_batch
        )

        # reset hack variables
        self._x_tree_mess = None
        self._x_jtmpn_holder = None

        # Log accuracies
        acc_dict = dict(word=1 - word_acc, topo=1 - topo_acc, assm=1 - assm_acc)
        if self.logging_prefix is not None:
            for k, v in acc_dict.items():
                self.log(f"{k}/{self.logging_prefix}", v)

        # Return sum of losses
        return word_loss + topo_loss + assm_loss

    def decode_deterministic(self, z: torch.Tensor) -> List[Any]:
        z_tree = z[:, :self.jtnn_vae.latent_T_size]
        z_mol = z[:, self.jtnn_vae.latent_T_size:]
        return self.jtnn_vae.decode(z_tree, z_mol, False)

    def training_step(self, batch, batch_idx):
        try:
            return super().training_step(batch, batch_idx, self.training_m, self.training_M)
        except RuntimeError:
            return utils._get_zero_grad_tensor(self.device)

    def validation_step(self, batch, batch_idx):
        try:
            return super().validation_step(batch, batch_idx, self.validation_m, self.validation_M)
        except RuntimeError:
            return utils._get_zero_grad_tensor(self.device)

    def backward(self, *args, **kwargs):
        try:
            super().backward(*args, **kwargs)
        except RuntimeError:
            pass
