import argparse
from TransVAE.transvae.trans_models import TransVAE
from TransVAE.transvae.trans_models_old import TransVAE as OldTransVAE
from TransVAE.transvae.rnn_models import RNN, RNNAttn

def model_init(args, params={}):
    ### Model Name
    if args.save_name is None:
        if args.model == 'transvae':
            save_name = 'trans{}x-{}_{}'.format(args.d_feedforward // args.d_model,
                                                args.d_model,
                                                args.data_source)
        else:
            save_name = '{}-{}_{}'.format(args.model,
                                          args.d_model,
                                          args.data_source)
    else:
        save_name = args.save_name

    ### Load Model
    if args.model == 'transvae':
        vae = TransVAE(params=params, name=save_name, d_model=args.d_model,
                       d_ff=args.d_feedforward, d_latent=args.d_latent,
                       property_predictor=args.property_predictor, d_pp=args.d_property_predictor,
                       depth_pp=args.depth_property_predictor,
                       metric_learning=args.metric_learning, metric=args.metric, threshold=args.threshold,
                       weighted_training=args.weighted_training,
                       )
    elif args.model == 'rnnattn':
        vae = RNNAttn(params=params, name=save_name, d_model=args.d_model,
                      d_latent=args.d_latent, property_predictor=args.property_predictor,
                      d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)
    elif args.model == 'rnn':
        vae = RNN(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)

    return vae

def model_init_old(args, params={}):
    ### Model Name
    if args.save_name is None:
        if args.model == 'transvae':
            save_name = 'trans{}x-{}_{}'.format(args.d_feedforward // args.d_model,
                                                args.d_model,
                                                args.data_source)
        else:
            save_name = '{}-{}_{}'.format(args.model,
                                          args.d_model,
                                          args.data_source)
    else:
        save_name = args.save_name

    ### Load Model
    if args.model == 'transvae':
        vae = OldTransVAE(params=params, name=save_name, d_model=args.d_model,
                       d_ff=args.d_feedforward, d_latent=args.d_latent,
                       property_predictor=args.property_predictor, d_pp=args.d_property_predictor,
                       depth_pp=args.depth_property_predictor)
    elif args.model == 'rnnattn':
        vae = RNNAttn(params=params, name=save_name, d_model=args.d_model,
                      d_latent=args.d_latent, property_predictor=args.property_predictor,
                      d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)
    elif args.model == 'rnn':
        vae = RNN(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)

    return vae


def train_parser():
    parser = argparse.ArgumentParser()
    ### Architecture Parameters
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--src_len', default=12, type=int)
    parser.add_argument('--tgt_len', default=11, type=int)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_feedforward', default=128, type=int)
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--property_predictor', default=False, action='store_true')
    parser.add_argument('--d_property_predictor', default=256, type=int)
    parser.add_argument('--depth_property_predictor', default=2, type=int)
    parser.add_argument('--metric_learning', default=False, action='store_true')
    parser.add_argument('--metric', default='constrastive', type=str, choices=['contrastive', 'triplet', 'logratio'])
    parser.add_argument('--threshold', default=0.1, type=float)
    ### Hyperparameters
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_chunks', default=5, type=int)
    parser.add_argument('--beta', default=0.05, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float)
    parser.add_argument('--anneal_start', default=0, type=int)
    parser.add_argument('--adam_lr', default=3e-4, type=float)
    parser.add_argument('--lr_scale', default=1, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--eps_scale', default=1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weighted_training', action='store_true', default=False)
    ### Data Parameters
    parser.add_argument('--data_source', choices=['zinc', 'pubchem', 'custom'],
                        required=True, type=str)
    parser.add_argument('--train_mols_path', default=None, type=str)
    parser.add_argument('--test_mols_path', default=None, type=str)
    parser.add_argument('--train_props_path', default=None, type=str)
    parser.add_argument('--test_props_path', default=None, type=str)
    parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--char_weights_path', default=None, type=str)
    ### Load Parameters
    parser.add_argument('--checkpoint', default=None, type=str)
    ### Save Parameters
    parser.add_argument('--save_name', default=None, type=str)
    parser.add_argument('--save_freq', default=5, type=int)

    return parser

def sample_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', default=None, type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy'],
                        required=True, type=str)
    parser.add_argument('--k', default=15, type=int)
    parser.add_argument('--condition', default='', type=str)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--n_samples', default=30000, type=int)
    parser.add_argument('--n_samples_per_batch', default=100, type=int)
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)
    ### Make latent space plot with labels
    parser.add_argument('--make_plot', action='store_true')
    parser.add_argument('--props', default='', type=str)

    return parser


def reconstruct_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', default=None, type=str)
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)
    ### Make latent space plot with labels
    parser.add_argument('--make_plot', action='store_true')
    parser.add_argument('--props', default='', type=str)

    return parser


def attn_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', required=True, type=str)
    ### Sampling Parameters
    parser.add_argument('--n_samples', default=5000, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_chunks', default=5, type=int)
    parser.add_argument('--shuffle', default=False, action='store_true')
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser

def vocab_parser():
    parser = argparse.ArgumentParser()
    ### Vocab Parameters
    parser.add_argument('--mols', required=True, type=str)
    parser.add_argument('--freq_penalty', default=0.5, type=float)
    parser.add_argument('--pad_penalty', default=0.1, type=float)
    parser.add_argument('--vocab_name', default='custom_char_dict', type=str)
    parser.add_argument('--weights_name', default='custom_char_weights', type=str)
    parser.add_argument('--save_dir', default='data', type=str)

    return parser
