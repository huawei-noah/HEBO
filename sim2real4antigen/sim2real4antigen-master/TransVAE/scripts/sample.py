import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
ROOTDIR = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path.insert(0, ROOTDIR)

from TransVAE.transvae.trans_models import TransVAE
from TransVAE.transvae.rnn_models import RNN, RNNAttn
from TransVAE.transvae.tvae_util import calc_entropy
from TransVAE.scripts.parsers import sample_parser

def sample(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    ### Parse conditional string
    if args.condition == '':
        condition = []
    else:
        condition = args.condition.split(',')

    ### Calculate entropy depending on sampling mode
    if args.sample_mode == 'rand':
        sample_mode = 'rand'
        sample_dims = None
    else:
        entropy_data = pd.read_csv(args.mols).to_numpy()
        if args.props != '':
            try:
                props = pd.read_csv(args.props).values.flatten()
            except UnicodeDecodeError:
                props = np.load(args.props).flatten()
                mean = np.load('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/props_mean.npy')
                std = np.load('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/props_std.npy')
                props = (props - mean) / std
            if len(entropy_data) > len(props):
                entropy_data = entropy_data[:len(props)]
                print(f'Using first {len(props)} points')
            else:
                props = props[:len(entropy_data)]
                print(f'Using first {len(entropy_data)} points')

        mems, mus, logvars = vae.calc_mems(entropy_data, log=False, save=False)
        vae_entropy = calc_entropy(mus)
        entropy_idxs = np.where(np.array(vae_entropy) > args.entropy_cutoff)[0]
        sample_dims = entropy_idxs
        if args.sample_mode == 'high_entropy':
            sample_mode = 'top_dims'
        elif args.sample_mode == 'k_high_entropy':
            sample_mode = 'k_dims'

    ### Generate samples
    samples = []
    n_gen = args.n_samples
    while n_gen > 0:
        current_samples = vae.sample(args.n_samples_per_batch, sample_mode=sample_mode,
                                     sample_dims=sample_dims, k=args.k, condition=condition)
        samples.extend(current_samples)
        n_gen -= len(current_samples)

    samples = pd.DataFrame(samples, columns=['mol'])
    if args.save_path is None:
        os.makedirs('generated', exist_ok=True)
        save_path = 'generated/{}_{}.csv'.format(vae.name, args.sample_mode)
    else:
        save_path = os.path.join(args.save_path, vae.name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{args.props.split("/")[-1].split(".")[0]}-{args.sample_mode}-'
                                            f'{args.model_ckpt.split("/")[-1].split(".")[0]}_ckpt.csv')
    samples.to_csv(save_path, index=False)

    if args.make_plot:
        if args.props != '':
            fig, ax = plt.subplots()
            im = ax.scatter(mems[:, 0], mems[:, 1], c=props, alpha=0.25, marker='.', label='Absolut! Energy')
            fig.colorbar(im, ax=ax)
            plt.legend()
            plt.title(f'Latent Space of {vae.name}')
            # plt.title(f'{vae.name} Latent Space of {args.mols.split("/")[-1].split(".")[0]}')
            plt.savefig(save_path.split(".")[0] + ".pdf")
            plt.close()
            print(f'Saved picture at', save_path.split(".")[0] + ".pdf")

        else:
            plt.scatter(mems[:, 0], mems[:, 1], color='C0', alpha=0.25, marker='.', label='Mem')
            plt.scatter(mus[:, 0], mus[:, 1], color='C1', alpha=0.25, marker='+', label='Mu')
            plt.legend()
            plt.title(f'CDR3 Seq. in Latent Space of {vae.name}')
            plt.savefig(save_path.split(".")[0] + ".pdf")
            # plt.show()
            plt.close()
        print(f'Saved picture at', save_path.split(".")[0] + ".pdf")



if __name__ == '__main__':
    parser = sample_parser()
    args = parser.parse_args()
    sample(args)
