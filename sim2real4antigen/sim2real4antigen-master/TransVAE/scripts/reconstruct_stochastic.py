import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
ROOTDIR = str(Path(os.path.realpath(__file__)).parent.parent.parent)
print('os.getcwd()', os.getcwd())
print('ROOTDIR', ROOTDIR)
sys.path.insert(0, ROOTDIR)

from TransVAE.transvae.trans_models import TransVAE
from TransVAE.transvae.rnn_models import RNN, RNNAttn
from TransVAE.scripts.parsers import reconstruct_parser

def reconstruct(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    data = pd.read_csv(args.mols).to_numpy()
    data = data[0].reshape(-1, 1)
    try:
        props = pd.read_csv(args.props).values.flatten()
    except UnicodeDecodeError:
        props = np.load(args.props).flatten()
        mean = np.load('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/props_mean.npy')
        std = np.load('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/props_std.npy')
        props = (props - mean) / std
    if len(data) > len(props):
        data = data[:len(props)]
        print(f'Using first {len(props)} points')
    else:
        props = props[:len(data)]
        print(f'Using first {len(data)} points')

    ### Calculate memories (i.e. encode data)
    smiles, mems = vae.reconstruct(data, n_samples=10, method='stochastic', log=False, return_mems=True, return_str=True)
    smiles_df = pd.DataFrame(smiles, columns=['mol'])
    if args.save_path is None:
        os.makedirs('generated', exist_ok=True)
        save_path_csv = 'generated/{}_{}.csv'.format(vae.name, args.sample_mode)
        save_path_txt = 'generated/{}_{}.txt'.format(vae.name, args.sample_mode)
    else:
        save_path = os.path.join(args.save_path, vae.name)
        os.makedirs(save_path, exist_ok=True)
        save_path_csv = os.path.join(save_path, f'{args.props.split("/")[-1].split(".")[0]}-'
                                            f'{args.model_ckpt.split("/")[-1].split(".")[0]}_ckpt.csv')
        save_path_txt = os.path.join(save_path, f'{args.props.split("/")[-1].split(".")[0]}-'
                                            f'{args.model_ckpt.split("/")[-1].split(".")[0]}_ckpt.txt')
    smiles_df.to_csv(save_path_csv, index=False)
    with open(save_path_txt, 'w') as f:
        f.write(f'0\t{data.item()}\n')
        for i, s in enumerate(smiles):
            f.write(f'{i+1}\t{s}\n')

    absolut_result = pd.read_csv(os.path.join(save_path, '1S78_BFinalBindings_Process_1_Of_1.txt'), sep='\t', header=1)
    energy = absolut_result['Energy'].values
    # plt.scatter(mems[:, 0], mems[:, 1], c=props, alpha=0.25, marker='.', label='Absolut! Energy')
    # # plt.scatter(mus[:, 0], mus[:, 1], color='C1', alpha=0.25, marker='+', label='Mu')
    # plt.legend()
    # plt.title(f'{vae.name} Latent Space of {args.mols.split("/")[-1].split(".")[0]}')
    # # plt.title(f'{args.mols.split("/")[-1].split(".")[0]} CDR3 Seq. in Latent Space of {vae.name}')
    # plt.savefig(save_path.split(".")[0] + ".pdf")
    # plt.show()
    # plt.close()
    if args.make_plot:
        _, smiles_mus, _ = vae.calc_mems(np.array(smiles)[1:].reshape(-1, 1), save=False, drop_last=False)

        fig, ax = plt.subplots()
        # im = ax.scatter(mems[:, 0], mems[:, 1], c=energy[0], alpha=0.25, marker='.', label='Absolut! Energy')
        # im = ax.scatter(smiles_mus[:, 0], smiles_mus[:, 1], c=energy[1:], alpha=0.25, marker='.', label='Absolut! Energy')
        im = ax.scatter(smiles_mus, smiles_mus, c=energy, alpha=0.25, marker='.', label='Absolut! Energy')
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.title(f'Latent Space of {vae.name}')
        # plt.title(f'{vae.name} Latent Space of {args.mols.split("/")[-1].split(".")[0]}')
        plt.savefig(save_path.split(".")[0] + ".pdf")
        plt.close()
        print(f'Saved picture at', save_path.split(".")[0] + ".pdf")



if __name__ == '__main__':
    parser = reconstruct_parser()
    args = parser.parse_args()
    reconstruct(args)
