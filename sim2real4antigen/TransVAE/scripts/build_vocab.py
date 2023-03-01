import os
import pickle

import numpy as np
import pandas as pd

from TransVAE.transvae.tvae_util import *
from TransVAE.scripts.parsers import vocab_parser

def build_vocab(args):
    ### Build vocab dictionary
    print('building dictionary...')
    char_dict = {'<start>': 0}
    char_idx = 1
    mol_toks = []
    with open(args.mols, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line.lower() in ['smile', 'smiles', 'selfie', 'selfies']:
                pass
            else:
                mol = tokenizer(line)
                for tok in mol:
                    if tok not in char_dict.keys():
                        char_dict[tok] = char_idx
                        char_idx += 1
                    else:
                        pass
                mol.append('<end>')
                mol_toks.append(mol)
    char_dict['_'] = char_idx
    char_dict['<end>'] = char_idx + 1

    ### Write dictionary to file
    with open(os.path.join(args.save_dir, args.vocab_name+'.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)

    ### Set weights params
    del char_dict['<start>']
    params = {'MAX_LENGTH': 126,
              'NUM_CHAR': len(char_dict.keys()),
              'CHAR_DICT': char_dict}

    ### Calculate weights
    print('calculating weights...')
    char_weights = get_char_weights(mol_toks, params, freq_penalty=args.freq_penalty)
    char_weights[-2] = args.pad_penalty
    np.save(os.path.join(args.save_dir, args.weights_name+'.npy'), char_weights)


if __name__ == '__main__':
    parser = vocab_parser()
    args = parser.parse_args()
    build_vocab(args)
