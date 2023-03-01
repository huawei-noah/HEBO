import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from TransVAE.transvae.trans_models import TransVAE
from TransVAE.transvae.rnn_models import RNN, RNNAttn

from TransVAE.transvae.data import vae_data_gen, make_std_mask
from TransVAE.scripts.parsers import attn_parser

def calc_attention(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    if args.shuffle:
        data = pd.read_csv(args.mols).sample(args.n_samples).to_numpy()
    else:
        data = pd.read_csv(args.mols).to_numpy()
        data = data[:args.n_samples,:]

    ### Load data and prepare for iteration
    data = vae_data_gen(data, props=None, char_dict=vae.params['CHAR_DICT'])
    data_iter = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=0,
                                            pin_memory=False, drop_last=True)
    save_shape = len(data_iter)*args.batch_size
    chunk_size = args.batch_size // args.batch_chunks

    ### Prepare save path
    if args.save_path is None:
        os.makedirs('attn_wts', exist_ok=True)
        save_path = 'attn_wts/{}'.format(vae.name)
    else:
        save_path = args.save_path

    ### Calculate attention weights
    vae.model.eval()
    if args.model == 'transvae':
        self_attn = torch.empty((save_shape, 4, 4, 127, 127))
        src_attn = torch.empty((save_shape, 3, 4, 126, 127))
        for j, data in enumerate(data_iter):
            for i in range(args.batch_chunks):
                batch_data = data[i*chunk_size:(i+1)*chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if vae.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != vae.pad_idx).unsqueeze(-2)
                tgt = Variable(mols_data[:,:-1]).long()
                tgt_mask = make_std_mask(tgt, vae.pad_idx)

                # Run samples through model to calculate weights
                mem, mu, logvar, pred_len, self_attn_wts = vae.model.encoder.forward_w_attn(vae.model.src_embed(src), src_mask)
                probs, deconv_wts, src_attn_wts = vae.model.decoder.forward_w_attn(vae.model.tgt_embed(tgt), mem, src_mask, tgt_mask)

                # Save weights to torch tensors
                self_attn_wts += deconv_wts
                start = j*args.batch_size+i*chunk_size
                stop = j*args.batch_size+(i+1)*chunk_size
                for k in range(len(self_attn_wts)):
                    self_attn[start:stop,k,:,:,:] = self_attn_wts[k]
                for k in range(len(src_attn_wts)):
                    src_attn[start:stop,k,:,:,:] = src_attn_wts[k]

        np.save(save_path+'_self_attn.npy', self_attn.numpy())
        np.save(save_path+'_src_attn.npy', src_attn.numpy())

    elif args.model == 'rnnattn':
        attn = torch.empty((save_shape, 1, 1, 127, 127))
        for j, data in enumerate(data_iter):
            for i in range(args.batch_chunks):
                batch_data = data[i*chunk_size:(i+1)*chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if vae.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()

                # Run samples through model to calculate weights
                mem, mu, logvar, attn_wts = vae.model.encoder(vae.model.src_embed(src), return_attn=True)
                start = j*args.batch_size+i*chunk_size
                stop = j*args.batch_size+(i+1)*chunk_size
                attn[start:stop,0,0,:,:] = attn_wts

        np.save(save_path+'.npy', attn.numpy())


if __name__ == '__main__':
    parser = attn_parser()
    args = parser.parse_args()
    calc_attention(args)
