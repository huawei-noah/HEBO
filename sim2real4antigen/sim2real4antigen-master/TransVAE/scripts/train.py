import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch

import sys
from pathlib import Path
ROOTDIR = str(Path(os.path.realpath(__file__)).parent.parent.parent)
print('os.getcwd()', os.getcwd())
print('ROOTDIR', ROOTDIR)
sys.path.insert(0, ROOTDIR)

from TransVAE.scripts.parsers import model_init, model_init_old, train_parser

def train(args):
    ### Update beta init parameter
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        start_epoch = ckpt['epoch']
        total_epochs = start_epoch + args.epochs
        beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
        args.beta_init = beta_init

    ### Build params dict
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps}

    ### Load data, vocab and token weights
    if args.data_source == 'custom':
        assert args.train_mols_path is not None and args.test_mols_path is not None and args.vocab_path is not None,\
        "ERROR: Must specify files for train/test data and vocabulary"
        train_mols = pd.read_csv(args.train_mols_path).to_numpy()
        test_mols = pd.read_csv(args.test_mols_path).to_numpy()
        if args.property_predictor or args.metric_learning:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open(args.vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
        if args.char_weights_path is not None:
            char_weights = np.load(args.char_weights_path)
            params['CHAR_WEIGHTS'] = char_weights
    else:
        train_mols = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
        test_mols = pd.read_csv('data/{}_test.txt'.format(args.data_source)).to_numpy()
        if args.property_predictor:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open('data/char_dict_{}.pkl'.format(args.data_source), 'rb') as f:
            char_dict = pickle.load(f)
        char_weights = np.load('data/char_weights_{}.npy'.format(args.data_source))
        params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict

    ### Add args to params so we don't need to change signature of constructor when we add arguments
    params['src_len'] = args.src_len
    params['tgt_len'] = args.tgt_len
    params['kernel_size'] = args.kernel_size

    ### Train model
    if args.data_source == 'custom':
        vae = model_init(args, params)
    else:
        vae = model_init_old(args, params)
    if args.checkpoint is not None:
        vae.load(args.checkpoint)
    vae.train(train_mols, test_mols, train_props, test_props,
              epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':
    parser = train_parser()
    args = parser.parse_args()
    train(args)
