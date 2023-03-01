import os
import pickle

import numpy as np
import pandas as pd

import torch

from argparse import Namespace

import sys
from pathlib import Path
ROOT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path.insert(0, ROOT)

from TransVAE.scripts.parsers import model_init, model_init_old, train_parser


def load_transvae(path):
    ### Load state_dict
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    params = ckpt['params']
    if isinstance(params['CHAR_WEIGHTS'], torch.Tensor):
        params['CHAR_WEIGHTS'] = params['CHAR_WEIGHTS'].cpu().numpy()

    # Create args
    non_capital_keys = [k for k in params if k.upper() != k]
    args_dict = {k: params[k] for k in non_capital_keys}
    args_dict['save_name'] = ckpt['name']
    args_dict['model'] = 'transvae'
    args_dict['d_feedforward'] = args_dict.pop('d_ff')
    args_dict['d_property_predictor'] = args_dict.pop('d_pp')
    args_dict['depth_property_predictor'] = args_dict.pop('depth_pp')
    args_dict['weighted_training'] = args_dict.get('weighted_training', False)
    args = Namespace(**args_dict)

    ### Initialize and load model
    vae = model_init(args, params)
    vae.load(path)
    return vae