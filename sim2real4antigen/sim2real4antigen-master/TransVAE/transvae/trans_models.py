import os
import json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler

from TransVAE.transvae.tvae_util import *
from TransVAE.transvae.opt import NoamOpt
from TransVAE.transvae.data import vae_data_gen, make_std_mask
from TransVAE.transvae.loss import vae_loss, trans_vae_loss, trans_vae_fixed_len_loss


####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """

    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BATCH_CHUNKS' not in self.params.keys():
            self.params['BATCH_CHUNKS'] = 5
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'LR' not in self.params.keys():
            self.params['LR_SCALE'] = 1
        if 'WARMUP_STEPS' not in self.params.keys():
            self.params['WARMUP_STEPS'] = 10000
        if 'EPS_SCALE' not in self.params.keys():
            self.params['EPS_SCALE'] = 1
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
            self.pad_idx = self.params['CHAR_DICT']['_']
            if 'CHAR_WEIGHTS' in self.params.keys():
                self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
            else:
                self.params['CHAR_WEIGHTS'] = torch.ones(self.vocab_size - 1,
                                                         dtype=torch.float)  # TODO compute manually
        self.loss_func = vae_loss
        self.data_gen = vae_data_gen

        ### Sequence length hard-coded into model
        self.src_len = 12
        self.tgt_len = 11
        # self.src_len = self.params['src_len']
        # self.tgt_len = self.params['tgt_len']

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}
        self.loaded_from = None

    def save(self, state, fn, path='saved_models', use_name=True):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        if self.name is not None:
            path = path + "/" + self.name
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                # if self.name is not None:
                #     fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)

    def load(self, checkpoint_path):
        """
        Loads a saved model state

        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.loaded_from = checkpoint_path
        for k in self.current_state.keys():
            try:
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError:
                self.current_state[k] = None

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
            else:
                pass
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.pad_idx = self.params['CHAR_DICT']['_']
        self.build_model()
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

    def train(self, train_mols, val_mols, train_props=None, val_props=None,
              epochs=100, save=True, save_freq=None, log=True, log_dir='trials'):
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            train_props (np.array): Numpy array containing chemical property of
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        ### Prepare data iterators
        train_data = self.data_gen(train_mols, train_props, char_dict=self.params['CHAR_DICT'], src_len=self.src_len)
        val_data = self.data_gen(val_mols, val_props, char_dict=self.params['CHAR_DICT'], src_len=self.src_len)

        train_iter = DataLoader(train_data,
                                batch_size=self.params['BATCH_SIZE'],
                                shuffle=True, num_workers=0,
                                pin_memory=False, drop_last=True)
        val_iter = DataLoader(val_data,
                              batch_size=self.params['BATCH_SIZE'],
                              shuffle=True, num_workers=0,
                              pin_memory=False, drop_last=True)
        if self.params["weighted_training"]:
            train_weights = torch.softmax(train_data[:, -1].flatten(), dim=0)
            val_weights = torch.softmax(val_data[:, -1].flatten(), dim=0)
            train_weighted_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))
            val_weighted_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_weights))
            train_iter = DataLoader(train_data,
                                    batch_size=self.params['BATCH_SIZE'],
                                    num_workers=0,
                                    pin_memory=False, drop_last=True,
                                    sampler=train_weighted_sampler)
            val_iter = DataLoader(val_data,
                                  batch_size=self.params['BATCH_SIZE'],
                                  num_workers=0,
                                  pin_memory=False, drop_last=True,
                                  sampler=val_weighted_sampler)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']

        torch.backends.cudnn.benchmark = True

        ### Determine save frequency
        if save_freq is None:
            save_freq = epochs

        ### Setup log file
        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_' + self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write(
                    'epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,kld_loss,prop_mse_loss,metric_loss,run_time\n')
            log_file.close()

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                 epochs, self.params['ANNEAL_START'])

        ### Epoch loop
        print('Start Training')
        for epoch in range(epochs):
            ### Train Loop
            self.model.train()
            losses = []
            bces = []
            prop_mses = []
            klds = []
            metrics = []
            beta = kl_annealer(epoch)
            for j, data in enumerate(train_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                avg_metric_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i * self.chunk_size:(i + 1) * self.chunk_size, :]
                    mols_data = batch_data[:, :-1]
                    props_data = batch_data[:, -1]
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:, :-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, _, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        # true_len = src_mask.sum(dim=-1)
                        if self.params["metric_learning"]:
                            mem4ml = self.model.encoder.reparameterize(mu, logvar, self.model.encoder.eps_scale)
                        else:
                            mem4ml = None
                        loss, bce, _, kld, prop_mse, metricloss = trans_vae_fixed_len_loss(
                            src, x_out, mu, logvar, true_prop, pred_prop,
                            self.params['CHAR_WEIGHTS'], beta,
                            metric_learning=self.params['metric_learning'],
                            metric=self.params.get('metric', None),
                            threshold=self.params.get('threshold', 0.1),
                            embs=mem4ml,
                        )
                        # avg_bcemask_losses.append(bce_mask.item())
                        avg_metric_losses.append(metricloss.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                avg_metric = np.mean(avg_metric_losses)
                losses.append(avg_loss)
                bces.append(avg_bce)
                klds.append(avg_kld)
                prop_mses.append(avg_prop_mse)
                metrics.append(avg_metric)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                            j, 'train',
                                                                            avg_loss,
                                                                            avg_bce,
                                                                            avg_bcemask,
                                                                            avg_kld,
                                                                            avg_prop_mse,
                                                                            avg_metric,
                                                                            run_time))
                    log_file.close()
            train_loss = np.mean(losses)
            train_bce = np.mean(bces)
            train_kld = np.mean(klds)
            train_prop_mse = np.mean(prop_mses)
            train_metric = np.mean(metrics)

            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                avg_metric_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i * self.chunk_size:(i + 1) * self.chunk_size, :]
                    mols_data = batch_data[:, :-1]
                    props_data = batch_data[:, -1]
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:, :-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:, -1])

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, _, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        # true_len = src_mask.sum(dim=-1)
                        if self.params["metric_learning"]:
                            mem4ml = self.model.encoder.reparameterize(mu, logvar, self.model.encoder.eps_scale)
                        else:
                            mem4ml = None
                        loss, bce, _, kld, prop_mse, metricloss = trans_vae_fixed_len_loss(
                            src, x_out, mu, logvar, true_prop, pred_prop,
                            self.params['CHAR_WEIGHTS'], beta,
                            metric_learning=self.params['metric_learning'],
                            metric=self.params.get('metric', None),
                            threshold=self.params.get('threshold', 0.1),
                            embs=mem4ml,
                        )
                        # avg_bcemask_losses.append(bce_mask.item())
                        avg_metric_losses.append(metricloss.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                avg_metric = np.mean(avg_metric_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'test',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_bcemask,
                                                                         avg_kld,
                                                                         avg_prop_mse,
                                                                         avg_metric,
                                                                         run_time))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)
            # print('Epoch - {} Train - {} Val - {} KLBeta - {}'.format(self.n_epochs, train_loss, val_loss, beta))
            print(f"[{self.n_epochs}/{epochs}] Train: {train_loss:.3f} | Test: {val_loss:.3f} | KLBeta: {beta:.6f}"
                  f" | BCE: {train_bce:.3f} | KLD: {train_kld:.3f} | PropMSE: {train_prop_mse:.3f}"
                  f" | MetricL: {train_metric:.3f}")

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                if save:
                    self.save(self.current_state, 'best')

            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str)

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.params['d_latent'])
        else:
            assert sample_dims is not None, "ERROR: Must provide sample dimensions"
            if mode == 'top_dims':
                z = torch.zeros((size, self.params['d_latent']))
                for d in sample_dims:
                    z[:, d] = torch.randn(size)
            elif mode == 'k_dims':
                z = torch.zeros((size, self.params['d_latent']))
                d_select = np.random.choice(sample_dims, size=k, replace=False)
                for d in d_select:
                    z[:, d] = torch.randn(size)
        return z

    def greedy_decode(self, mem, src_mask=None, condition=[]):
        """
        Greedy decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0], 1).fill_(start_symbol).long()
        for tok in condition:
            condition_symbol = self.params['CHAR_DICT'][tok]
            condition_vec = torch.ones(mem.shape[0], 1).fill_(condition_symbol).long()
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0], max_len + 1).fill_(start_symbol).long()
        tgt[:, :len(condition) + 1] = decoded
        if src_mask is None and self.model_type == 'transformer':
            if self.params.get('pred_len', False):
                mask_lens = self.model.encoder.predict_mask_length(mem)
            else:
                mask_lens = np.ones((mem.shape[0], 1), dtype=int) * self.src_len
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len + 1))
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i, :, :mask_len] = torch.ones((1, 1, mask_len))
        elif self.model_type != 'transformer':
            src_mask = torch.ones((mem.shape[0], 1, self.src_len))

        if self.use_gpu:
            src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()

        self.model.eval()
        for i in range(len(condition), max_len):
            if self.model_type == 'transformer':
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                if self.use_gpu:
                    decode_mask = decode_mask.cuda()
                out = self.model.decode(mem, src_mask, Variable(decoded), decode_mask)
            else:
                out, _ = self.model.decode(tgt, mem)
            out = self.model.generator(out)
            prob = F.softmax(out[:, i, :], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            tgt[:, i + 1] = next_word
            if self.model_type == 'transformer':
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
        decoded = tgt[:, 1:]
        return decoded

    def stochastic_decode(self, mem, n_samples=1, src_mask=None, condition=None):
        """
        Stochastic decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        if condition is None:
            condition = []

        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0], n_samples, 1).fill_(start_symbol).long()
        for tok in condition:
            condition_symbol = self.params['CHAR_DICT'][tok]
            condition_vec = torch.ones(mem.shape[0], 1).fill_(condition_symbol).long()
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0], n_samples, max_len + 1).fill_(start_symbol).long()
        tgt[:, :, :len(condition) + 1] = decoded
        if src_mask is None and self.model_type == 'transformer':
            if self.params.get('pred_len', False):
                mask_lens = self.model.encoder.predict_mask_length(mem)
            else:
                mask_lens = np.ones((mem.shape[0], 1), dtype=int) * self.src_len
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len + 1))
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i, :, :mask_len] = torch.ones((1, 1, mask_len))
        elif self.model_type != 'transformer':
            src_mask = torch.ones((mem.shape[0], 1, self.src_len))

        if self.use_gpu:
            src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()

        self.model.eval()
        for i in range(len(condition), max_len):
            if self.model_type == 'transformer':
                decode_mask = Variable(subsequent_mask(decoded.size(-1)).long())
                # decode_mask = subsequent_mask(decoded.size(-1)).long()
                if self.use_gpu:
                    decode_mask = decode_mask.cuda()

                outs = []
                for j in range(n_samples):
                    logits = self.model.decode(mem, src_mask, Variable(decoded)[:, j, :], decode_mask)
                    outs.append(logits[:, -1, :])
                out = torch.cat(outs).unsqueeze(0)
            else:
                out, _ = self.model.decode(tgt, mem)

            out = self.model.generator(out)
            prob = F.softmax(out, dim=-1)
            prob[:, :, -2:] = 0.
            prob /= prob.sum(-1).unsqueeze(-1)
            cat = torch.distributions.Categorical(probs=prob)
            next_word = cat.sample()
            # np.random.choice()
            # _, next_word = torch.max(prob, dim=1)
            next_word += 1
            tgt[:, :, i + 1] = next_word.flatten()
            if self.model_type == 'transformer':
                next_word = next_word.unsqueeze(0).transpose(-1, -2)
                decoded = torch.cat([decoded, next_word], dim=-1)
        decoded = tgt[:, :, 1:]
        return decoded

    def reconstruct(self, data, method='greedy', n_samples=1, log=True, return_mems=True, return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'], src_len=self.src_len)

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=False)
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']

        self.model.eval()
        decoded_smiles = []
        mems = torch.empty((data.shape[0], self.params['d_latent'])).cpu()
        print('Reconstruct')
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('calcs/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i * self.chunk_size:(i + 1) * self.chunk_size, :]
                if len(batch_data) == 0:
                    break
                mols_data = batch_data[:, :-1]
                props_data = batch_data[:, -1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    _, mem, _, _ = self.model.encode(src, src_mask)  # technically this is mu
                else:
                    _, mem, _ = self.model.encode(src)
                start = j * self.batch_size + i * self.chunk_size
                stop = j * self.batch_size + (i + 1) * self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()

                ### Decode logic
                if method == 'greedy':
                    decoded = self.greedy_decode(mem, src_mask=src_mask)
                elif method == 'stochastic':
                    decoded = self.stochastic_decode(mem, n_samples=n_samples, src_mask=src_mask)
                else:
                    decoded = None

                if return_str:
                    decoded = decode_mols(decoded, self.params['ORG_DICT'], fast=method=='stochastic')
                    decoded_smiles += decoded
                else:
                    decoded_smiles.append(decoded)

        if return_mems:
            return decoded_smiles, mems.detach().numpy()
        else:
            return decoded_smiles

    def sample(self, n, method='greedy', sample_mode='rand',
               sample_dims=None, k=None, return_str=True,
               condition=[], return_z: bool = False):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
            return_z (bool): If true, returns the latent sampled from memory along with sampled
            sequences.
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k=k)

        if self.use_gpu:
            mem = mem.cuda()

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem, condition=condition)
        elif method == 'entropy':
            decoded = self.entropy_decode(mem, condition=condition)  # TODO
        else:
            decoded = None

        if return_str:
            decoded = decode_mols(decoded, self.params['ORG_DICT'])

        if return_z:
            return decoded, mem
        else:
            return decoded

    def calc_mems(self, data, log=True, save_dir='memory', save_fn='model_name', save=True, drop_last=True):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'], src_len=self.src_len)

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=drop_last)
        save_shape = len(data_iter) * self.params['BATCH_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']
        mems = torch.empty((save_shape, self.params['d_latent'])).cpu()
        mus = torch.empty((save_shape, self.params['d_latent'])).cpu()
        logvars = torch.empty((save_shape, self.params['d_latent'])).cpu()

        self.model.eval()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('memory/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i * self.chunk_size:(i + 1) * self.chunk_size, :]
                mols_data = batch_data[:, :-1]
                props_data = batch_data[:, -1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    mem, mu, logvar, _ = self.model.encode(src, src_mask)
                else:
                    mem, mu, logvar = self.model.encode(src)
                start = j * self.batch_size + i * self.chunk_size
                stop = j * self.batch_size + (i + 1) * self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()
                mus[start:stop, :] = mu.detach().cpu()
                logvars[start:stop, :] = logvar.detach().cpu()

        if save:
            if save_fn == 'model_name':
                save_fn = self.name
            save_path = os.path.join(save_dir, save_fn)
            np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
            np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
            np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
        else:
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()


####### Encoder, Decoder and Generator ############

class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """

    def __init__(self, params={}, name=None, N=3, d_model=128, d_ff=512,
                 d_latent=128, h=4, dropout=0.1, bypass_bottleneck=False,
                 property_predictor=False, d_pp=256, depth_pp=2,
                 metric_learning=False, metric='', threshold=0.1,
                 weighted_training=False,
                 load_fn=None):
        super().__init__(params=params, name=name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_ff (int): Dimensionality of feed-forward layers
            d_latent (int): Dimensionality of latent space
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
            property_predictor (bool): If true, model will predict property from latent memory
            d_pp (int): Dimensionality of property predictor layers
            depth_pp (int): Number of property predictor layers
            metric_learning (bool): If true, use metric loss in training
            metric (str): Type of metric to use in metric loss
            threshold (float): Threshold parameter to use for contrastive and triplet metric loss
            weighted_training (bool): If true, use property values to weight dataloader samples for train/val
            load_fn (str): Path to checkpoint file
        """

        ### Store architecture params
        self.model_type = 'transformer'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_ff'] = d_ff
        self.params['d_latent'] = d_latent
        self.params['h'] = h
        self.params['dropout'] = dropout
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.params["metric_learning"] = metric_learning
        self.params["metric"] = metric
        self.params["threshold"] = threshold
        self.params["weighted_training"] = weighted_training
        self.arch_params = ['N', 'd_model', 'd_ff', 'd_latent', 'h', 'dropout', 'bypass_bottleneck',
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
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.params['h'], self.params['d_model'])
        ff = PositionwiseFeedForward(self.params['d_model'], self.params['d_ff'], self.params['dropout'])
        position = PositionalEncoding(self.params['d_model'], self.params['dropout'])
        encoder = VAEEncoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                             self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'],
                             self.params['EPS_SCALE'], pred_len=self.params.get('pred_len', False))
        decoder = VAEDecoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                             DecoderLayer(self.params['d_model'], self.tgt_len, c(attn), c(attn), c(ff),
                                          self.params['dropout']),
                             self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'])
        src_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        tgt_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        generator = Generator(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'],
                                                   self.params['d_latent'])
        else:
            property_predictor = None
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator, property_predictor)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = NoamOpt(self.params['d_model'], self.params['LR_SCALE'], self.params['WARMUP_STEPS'],
                                 torch.optim.Adam(self.model.parameters(), lr=0,
                                                  betas=(0.9, 0.98), eps=1e-9))


class EncoderDecoder(nn.Module):
    """
    Base transformer Encoder-Decoder architecture
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, property_predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences"
        mem, mu, logvar, pred_len = self.encode(src, src_mask)
        x = self.decode(mem, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mu)
        else:
            prop = None
        return x, mu, logvar, pred_len, prop

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)

    def predict_property(self, mu):
        return self.property_predictor(mu)


class Generator(nn.Module):
    "Generates token predictions after final decoder layer"

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab - 1)

    def forward(self, x, mask=None):
        if mask is None:
            return self.proj(x)
        else:
            proj = self.proj(x)
            return proj * mask


class VAEEncoder(nn.Module):
    "Base transformer encoder architecture"

    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale, kernel_size=None, pred_len=True):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size, kernel_size=kernel_size)
        self.z_means = nn.Linear(layer.size * 2 * self.conv_bottleneck.kernel_size, d_latent)
        self.z_var = nn.Linear(layer.size * 2 * self.conv_bottleneck.kernel_size, d_latent)
        # self.z_means, self.z_var = nn.Linear(576, d_latent), nn.Linear(576, d_latent)
        self.norm = LayerNorm(layer.size)
        # No need to predict length of sequence in our case they are all the same size
        self._pred_len = pred_len
        if self._pred_len:
            self.predict_len1 = nn.Linear(d_latent, d_latent * 2)
            self.predict_len2 = nn.Linear(d_latent * 2, d_latent)

        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        if self._pred_len:
            pred_len = self.predict_len1(mem)
            pred_len = self.predict_len2(pred_len)
            pred_len = F.softmax(pred_len, dim=-1)
            pred_len = torch.topk(pred_len, 1)[1]
            return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        "Stochastic reparameterization"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps * std

    def forward(self, x, mask):
        ### Attention and feedforward layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        mem = self.norm(x)
        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
            pred_len = None
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            if self._pred_len:
                pred_len = self.predict_len1(mu)
                pred_len = self.predict_len2(pred_len)
            else:
                pred_len = None
        return mem, mu, logvar, pred_len

    def forward_w_attn(self, x, mask):
        "Forward pass that saves attention weights"
        attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mask, return_attn=True)
            attn_wts.append(wts.detach().cpu())
        mem = self.norm(x)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len, attn_wts


class EncoderLayer(nn.Module):
    "Self-attention/feedforward implementation"

    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)


class VAEDecoder(nn.Module):
    "Base transformer decoder architecture"

    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck, kernel_size=None):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len

        # Reshaping memory with deconvolution
        self.linear = nn.Linear(d_latent, 256)
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size, kernel_size=kernel_size)

    def forward(self, x, mem, src_mask, tgt_mask):
        ### Deconvolutional bottleneck (up-sampling)
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            # mem = mem.view(-1, 64, 4)
            # mem = mem.view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        ### Final self-attention layer
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        # Batch normalization
        mem = self.norm(mem)
        ### Source-attention layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        return self.norm(x)

    def forward_w_attn(self, x, mem, src_mask, tgt_mask):
        "Forward pass that saves attention weights"
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem, deconv_wts = final_encode(mem, src_mask, return_attn=True)
        mem = self.norm(mem)
        src_attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mem, mem, src_mask, tgt_mask, return_attn=True)
            src_attn_wts.append(wts.detach().cpu())
        return self.norm(x), [deconv_wts.detach().cpu()], src_attn_wts


class DecoderLayer(nn.Module):
    "Self-attention/source-attention/feedforward implementation"

    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask, return_attn=False):
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)


############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """

    def __init__(self, size, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 4
        self.kernel_size = kernel_size
        conv_layers = [
            nn.Conv1d(size, int(size * 1.3), kernel_size),
            nn.Conv1d(int(size * 1.3), int(size * 1.6), kernel_size),
            nn.Conv1d(int(size * 1.6), size * 2, kernel_size)
        ]
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x


class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """

    def __init__(self, size, kernel_size=4):
        super().__init__()
        if kernel_size is None:
            kernel_size = 4
        self.kernel_size = kernel_size
        self._size = size
        deconv_layers = [
            nn.Sequential(nn.ConvTranspose1d(size * 2, int(size * 1.6), kernel_size, stride=2, padding=2)),
            nn.Sequential(nn.ConvTranspose1d(int(size * 1.6), int(size * 1.3), kernel_size, stride=2, padding=2)),
            nn.Sequential(nn.ConvTranspose1d(int(size * 1.3), size, kernel_size, stride=1, padding=0))
        ]
        # deconv_layers = [
        #     nn.Sequential(nn.ConvTranspose1d(size * 2, int(size * 1.6), kernel_size, stride=2, padding=2)),
        #     nn.Sequential(nn.ConvTranspose1d(int(size * 1.6), int(size * 1.3), kernel_size, stride=2, padding=2)),
        #     nn.Sequential(nn.ConvTranspose1d(int(size * 1.3), size, kernel_size + 1, stride=1, padding=1))
        # ]
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(x.shape[0], self._size * 2, -1)
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x


# class ConvBottleneck(nn.Module):
#     """
#     Set of convolutional layers to reduce memory matrix to single
#     latent vector
#     """
#     def __init__(self, size, kernel_size=None):
#         super().__init__()
#         conv_layers = []
#         in_d = size
#         first = True
#         for i in range(3):
#             out_d = int((in_d - 64) // 2 + 64)
#             if first:
#                 kernsize = 9 if kernel_size is None else kernel_size + 1
#                 first = False
#             else:
#                 kernsize = 8 if kernel_size is None else kernel_size
#             if i == 2:
#                 out_d = 64
#             conv_layers.append(nn.Conv1d(in_d, out_d, kernsize))
#             # conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernsize), nn.MaxPool1d(2)))
#             in_d = out_d
#         self.conv_layers = ListModule(*conv_layers)
#
#     def forward(self, x):
#         for conv in self.conv_layers:
#             x = F.relu(conv(x))
#         return x
#
# class DeconvBottleneck(nn.Module):
#     """
#     Set of deconvolutional layers to reshape latent vector
#     back into memory matrix
#     """
#     def __init__(self, size, kernel_size=None):
#         super().__init__()
#         deconv_layers = []
#         in_d = 64
#         for i in range(3):
#             out_d = (size - in_d) // 4 + in_d
#             stride = 1
#             # stride = 4 - i
#             kernsize = 11 if kernel_size is None else kernel_size
#             if i == 2:
#                 out_d = size
#                 stride = 1
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernsize,
#                                                                   stride=stride, padding=2)))
#             in_d = out_d
#         self.deconv_layers = ListModule(*deconv_layers)
#
#     def forward(self, x):
#         for deconv in self.deconv_layers:
#             x = F.relu(deconv(x))
#         return x

############## Property Predictor #################

class PropertyPredictor(nn.Module):
    "Optional property predictor module"

    def __init__(self, d_pp, depth_pp, d_latent):
        super().__init__()
        prediction_layers = []
        for i in range(depth_pp):
            if i == 0:
                linear_layer = nn.Linear(d_latent, d_pp)
            elif i == depth_pp - 1:
                linear_layer = nn.Linear(d_pp, 1)
            else:
                linear_layer = nn.Linear(d_pp, d_pp)
            prediction_layers.append(linear_layer)
        self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x):
        for prediction_layer in self.prediction_layers:
            x = F.relu(prediction_layer(x))
        return x


############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (pytorch)"

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        return self.bn(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))
