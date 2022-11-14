import numpy as np
import torch
from dataloader.dataset import TransformerSeq2Seq
from torch.utils.data import DataLoader, random_split
import os
from abc import ABC


class MultiModalDataLoader(ABC):
    """"Class used to create train and test data loaders for either graph or standard feature vector inputs.
        """


    def __init__(self, config):
        data = {'seq_len': 0, 'batch_size': 64, 'nm_data_seq': 0,
                'nm_gen_seq': 250,
                'normalise': True,
                'path': './data',
                'target': '1ADQ',
                'filename': '_top_70000_corpus.csv'}

        model = {
            "nm_filters": 400,
            "seq_len": 43,
            "kernel_size": 3,
            "stride": 1,
            "drop_p": 0.2,
            "pool_size": 2,
            "pool_stride": 1,
            "affine": 300,
            "vocab_size": 21}

        seq2seq = {"vocab_size": 20,
                   "embedding_dim": 256,
                   "rnn_units": 1024,
                   "padding_idx": 20}

        optim = {
            "lr": 0.0075,
            "step_size": 200,
            "gamma": 0.1,
            "beta1": 0.5,
            "beta2": 0.999}

        self.config = {'optim': optim, 'model': model, 'seq2seq': seq2seq, 'data': data,
                       'model_path': "./data/models",
                       'results_path': "./data/results",
                       "gpu_id": 3,
                       "seed": 42,
                       "test_every": 1,
                       "save_every": 5,
                       "epochs": 100,
                       "test_epoch": 10,
                       "cuda": True,
                       "problem": "classification",
                       }

        useGraph = self.config.get('graph', False)
        if useGraph:
            self.GraphDataLoader()
        else:
            self.StandardDataLoader()


    def GraphDataLoader(self):
        return NotImplementedError


    def StandardDataLoader(self):
        # Create some directories
        if not os.path.exists(self.config['results_path']):
            os.makedirs(self.config['results_path'])
        if not os.path.exists(self.config['model_path']):
            os.makedirs(self.config['model_path'])

        dataset = Seq2SeqDataset(self.config['data'])

        self.config['model']['vocab_size'] = dataset.vocab_size

        nm_samples = len(dataset.data)
        train_samples = int(0.8 * nm_samples)
        test_samples = nm_samples - train_samples

        trainset, testset = random_split(dataset, [train_samples, test_samples],
                                         generator=torch.Generator().manual_seed(self.config['seed']))

        train_loader = DataLoader(trainset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=4,
                                  drop_last=True)
        valid_loader = DataLoader(testset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=4,
                                 drop_last=True)

        return train_loader, valid_loader


class SequenceDataLoader(ABC):
    """"Class used to create train and test data loaders for either CDR sequences
        config:
            path: Path to CDR3 data
            antigens: list of antigen to use, if None use all antigens
            nm_workers: Number of workers to use for dataloader
            seed: seed for random split
            test_size: fraction of samples used for testing
        """
    def __init__(self, config,
                 tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = TransformerSeq2Seq(self.config['path'],
                                            self.tokenizer,
                                            self.config['antibody'],
                                            self.config['antigens'],
                                            self.config['return_energy'])
        self.nm_samples = len(self.dataset.sequences)
        self.test_samples = int(self.config['test_size'] * self.nm_samples)
        self.train_samples = self.nm_samples - self.test_samples

    def StandardDataLoader(self):
        trainset, testset = random_split(self.dataset, [self.train_samples, self.test_samples],
                                         generator=torch.Generator().manual_seed(self.config['seed']))
        return trainset, testset
