import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from TransVAE.transvae.tvae_util import *
from TransVAE.transvae.opt import NoamOpt, AdamOpt
from TransVAE.transvae.trans_models import VAEShell, Generator, ConvBottleneck, DeconvBottleneck, PropertyPredictor, Embeddings, LayerNorm

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Attention architectures inspired by the ^^^ implementation
########## Model Classes ############

class RNNAttn(VAEShell):
    """
    RNN-based VAE class with attention.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_attention=False,
                 bypass_bottleneck=False,
                 property_predictor=False,
                 d_pp=256, depth_pp=2,
                 load_fn=None):
        super().__init__(params, name)
        """
        Instatiating a GruaVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_latent (int): Dimensionality of latent space
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
        """

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'rnn_attn'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.params['bypass_attention'] = bypass_attention
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force',
                            'bypass_attention', 'bypass_bottleneck', 'property_predictor',
                            'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = RNNAttnEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                                 self.params['dropout'], self.src_len, self.params['bypass_attention'],
                                 self.params['bypass_bottleneck'], self.device)
        decoder = RNNAttnDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                                 self.params['dropout'], self.params['teacher_force'], self.params['bypass_bottleneck'],
                                 self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)

class RNN(VAEShell):
    """
    RNN-based VAE without attention.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_bottleneck=False, property_predictor=False,
                 d_pp=256, depth_pp=2, load_fn=None):
        super().__init__(params, name)

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'rnn'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = RNNEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.params['bypass_bottleneck'], self.device)
        decoder = RNNDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], 125, self.params['teacher_force'], self.params['bypass_bottleneck'],
                             self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)


########## Recurrent Sub-blocks ############

class RNNEncoderDecoder(nn.Module):
    """
    Recurrent Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,
                 property_predictor, params):
        super().__init__()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        mem, mu, logvar = self.encode(src)
        x, h = self.decode(tgt, mem)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mu)
        else:
            prop = None
        return x, mu, logvar, prop

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, mem):
        return self.decoder(self.src_embed(tgt), mem)

    def predict_property(self, mu):
        return self.property_predictor(mu)

class RNNAttnEncoder(nn.Module):
    """
    Recurrent encoder with attention architecture
    """
    def __init__(self, size, d_latent, N, dropout, src_length, bypass_attention, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = src_length+1
        self.bypass_attention = bypass_attention
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.attn = nn.Linear(self.size * 2, self.max_length)
        self.conv_bottleneck = ConvBottleneck(size)
        self.z_means = nn.Linear(576, d_latent)
        self.z_var = nn.Linear(576, d_latent)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, return_attn=False):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x_out, h = self.gru(x, h)
        x = x.permute(1, 0, 2)
        x_out = x_out.permute(1, 0, 2)
        mem = self.norm(x_out)
        if not self.bypass_attention:
            attn_weights = F.softmax(self.attn(torch.cat((x, mem), 2)), dim=2)
            attn_applied = torch.bmm(attn_weights, mem)
            mem = F.relu(attn_applied)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([100.])), Variable(torch.tensor([100.]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar)
        if return_attn:
            return mem, mu, logvar, attn_weights.detach().cpu()
        else:
            return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)

class RNNAttnDecoder(nn.Module):
    """
    Recurrent decoder with attention architecture
    """
    def __init__(self, size, d_latent, N, dropout, tf, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.linear = nn.Linear(d_latent, 576)
        self.deconv_bottleneck = DeconvBottleneck(size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        embedded = self.dropout(tgt)
        h = self.initH(mem.shape[0])
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.contiguous().view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
            mem = self.norm(mem)
        mem = mem[:,:-1,:]
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()

class RNNEncoder(nn.Module):
    """
    Simple recurrent encoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.z_means = nn.Linear(size, d_latent)
        self.z_var = nn.Linear(size, d_latent)
        self.norm = LayerNorm(size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x, h = self.gru(x, h)
        mem = self.norm(h[-1,:,:])
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar)
        return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()

class RNNDecoder(nn.Module):
    """
    Simple recurrent decoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, tgt_length, tf, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = tgt_length+1
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.unbottleneck = nn.Linear(d_latent, size)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        h = self.initH(mem.shape[0])
        embedded = self.dropout(tgt)
        if not self.bypass_bottleneck:
            mem = F.relu(self.unbottleneck(mem))
            mem = mem.unsqueeze(1).repeat(1, self.max_length, 1)
            mem = self.norm(mem)
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)
