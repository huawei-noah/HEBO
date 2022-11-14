import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base import BaseModel
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


class CDR3VAE(BaseModel, Module):
    def __init__(self, config, device):
        BaseModel.__init__(self)
        Module.__init__(self)
        self.config = config
        self.device = device

        self.encoder.apply(self.weights_init(init_type=self.config['w_init']))
        self.decoder.apply(self.weights_int(init_type=self.config['w_init']))

        self.recon_loss = nn.NLLLoss(reduction='sum')

    def weights_init(self, init_type='Kaiming'):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Linear') == 0 and hasattr(m, 'weight'):
                if init_type == 'Gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'Xavier':
                    init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
                elif init_type == 'Kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, model='fan_in')
                elif init_type == 'Orthogonal':
                    init.orthogonal_(m.weight.data, gain=np.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, f"Initialisation not implemented {init_type}"
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        return init_func

    def sampling(self, h):
        mu, logsigma = self.encoder.q_mu(h), self.encoder.q_logsigma(h)
        z = mu + torch.randn(*h.shape).to(h.device) * torch.exp(0.5 * logsigma)
        return z, mu, logsigma

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logsigma = self.sampling(h)
        x_gen = self.decoder(z)
        return x_gen, z, mu, logsigma

    def kl_divergence(self, mu, logsigma):
        return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(2))

    def logpx(self, x, x_gen):
        x_gen = F.LogSoftmax(x_gen, dim=1)
        return self.recon_loss(x, x_gen)

    def save_model(self, epoch, itern_train, itern_test):
        model = f"{self.config['path']}/model_{epoch + 1}.pth"
        torch.save({'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict,
                    'optim': optim.state_dict(), 'itern_train': itern_train, 'itern_test': itern_test}, model)

    def resume(self, epoch, optim=None, device=torch.device('cpu')):
        ckpt = f"{self.config['path']}/model_{epoch + 1}.pth"
        state = torch.load(ckpt, map_location=device)
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        if optim:
            optim.load_state_dict(state['optim'])
            itern = state['itern']
            return optim, itern

    def fit(self):
        # Reproducibility of Experiments
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device(f"cuda:{self.config['gpu_id']}" if config['cuda'] else 'cpu')

        dataset = SequenceDataLoader(self.config['data'])
        trainset, testset = dataset.StandardDataLoader()

        train_loader = DataLoader(trainset, batch_size=self.config['batch_size'],
                                  shuffle=True,
                                  num_workers=self.config['nm_workers'],
                                  drop_last=True)
        valid_loader = DataLoader(testset, batch_size=self.config['batch_size'],
                                  shuffle=False,
                                  num_workers=self.config['nm_workers'],
                                  drop_last=True)

        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.config['optim']['lr'],
                                 betas=(self.config['optim']['beta1'],
                                        self.config['optim']['beta2']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0, last_epoch=-1)

        writer = SummaryWriter(f"{self.config['model']}/loss_history")

        if config['resume']:
            optim, itern_train, itern_test = self.resume(self.config['resume_epoch'], optim, device)
            start_epoch = self.config['resume_epoch']
        else:
            itern_train, itern_test, start_epoch = 0, 0, 0
        for epoch in range(start_epoch, self.config['epochs']):
            for _, batch_x in enumerate(train_loader):
                batch_x = batch_x.to(device)
                optim.zero_grad()
                x_gen, z, mu, logsigma = self.forward(batch_x)
                kld = self.kl_divergence(mu, logsigma)
                logp = self.logpx(x, x_gen)
                loss = kld + logp
                loss.backward()
                writer.add_scalar("Reconstruction_Loss", logp.item(), itern)
                writer.add_scalar("KL_Divergence", kld.item(), itern)
                writer.add_scalar("Full_ELBO", loss.item(), itern)
                optim.step()
                itern += 1
            scheduler.step()
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_model(epoch, itern_train, itern_test)

            if (epoch + 1) % self.config['test_every'] == 0:
                with torch.no_grad():
                    for _, batch_x in enumerate(valid_loader):
                        batch_x = batch_x.to(device)
                        x_gen, z, mu, logsigma = self.forward(batch_x)
                        kld = self.kl_divergence(mu, logsigma)
                        logp = self.logpx(x, x_gen)
                        loss = kld + logp
                        writer.add_scalar("Validation_Reconstruction_Loss:", logp.item(), itern_test)
                        writer.add_scalar("Validation_KL_Divergence", kld.item(), itern_test)
                        writer.add_scalar("Validation_Full_ELBO", loss.item(), itern_test)
                        itern_test += 1
        writer.close()


from pvae.utils import get_mean_param


class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
               means.view(-1, *means.size()[2:]), \
               samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset): pass
