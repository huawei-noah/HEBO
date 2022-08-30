import pdb

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import re
import os
def energy_to_labels(energy):
    mean = energy.mean()
    labels = 1*(energy>=mean)
    return np.asarray(labels)

class TransformerSeq2Seq(data.Dataset):
    def __init__(self,
                 path,
                 tokenizer=None,
                 antibody='Murine',
                 antigens=None,
                 return_energy=False):
        '''
        path: path to CDR Data
        antigens: list of antigens to use if None use all antigen
        '''
        self.path = path
        self.antigen = antigens
        self.tokenizer = tokenizer
        self.return_energy = return_energy
        if not self.antigen or len(self.antigen)==0:
            filenames = glob.glob(f"{self.path}/*.txt")
        else:
            filenames = [glob.glob(f"{self.path}/RawBindings{antibody}/{antigen}/*.txt") for antigen in self.antigen]
            filenames = [file for files in filenames for file in files]
        if self.tokenizer:
            self.sequences = self.preprocess_for_transformers(filenames)
        else:
            self.sequences, energy = self.get_slidesCDRs(filenames)
            if self.return_energy:
                self.energy = energy
        self.maxlen = 11
        #self.maxlen = max([len(seq) for seq in sequences])
        if not self.tokenizer:
            AAs = 'ACDEFGHIKLMNPQRSTVWY'
            self.AA_to_idx = {aa:i for i, aa in enumerate(AAs)}
            self.AA_to_idx['pad'] = len(AAs)
            self.idx_to_AA = {i:aa for aa, i in self.AA_to_idx.items()}
            self.sequences = self.sequence_to_idx(self.sequences)
        else:
            self.sequences = list(set(self.sequences))

    def preprocess_for_transformers(self, filenames):
        data = f"{self.path}/preprocessBERT.txt"
        if os.path.exists(data):
            data = np.array([line.strip().split('\t') for line in open(data,'r')])
            sequences = data[:,1]
        else:
            sequences = self.get_slidesCDRs(filenames)
            with open(data, 'w') as f:
                for i, sequence in enumerate(sequences):
                    f.write(f"{i+1}\t{sequence}\n")
        return sequences

    def get_slidesCDRs(self, filenames):
        for i in range(len(filenames)):
            try:
                if i == 0:
                    df = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                    df = list(set(df['Slide'].dropna().values))
                else:
                    df_i = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                    df_i = list(set(df_i['Slide'].dropna().values))
                    df = list(set(df + df_i))
            except pd.errors.ParserError as err:
                print(f"{filenames[i]} causes an error {err}")
                continue
        return df

    def sequence_to_idx(self, sequences):
        seq_to_idx = []
        for seq in sequences:
            aa2idx = [self.AA_to_idx[aa] for aa in seq]
            if len(seq)<self.maxlen:
                aa2idx += [self.AA_to_idx['pad']] * (self.maxlen-len(seq))
            seq_to_idx.append(aa2idx)
        return seq_to_idx

    def __getitem__(self, index):
        if not self.tokenizer:
            if self.return_energy:
                return torch.tensor(self.sequences[index]), self.energy[index]
            return torch.tensor(self.sequences[index])
        if torch.is_tensor(index):
            index = index.tolist()
        seq = " ".join(self.sequences[index])
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_to_idx = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.maxlen)
        # seq_to_idx = {k:torch.tensor(v) for k,v in seq_to_idx.items()}
        seq_to_idx["labels"] = seq_to_idx["input_ids"].copy()
        return seq_to_idx

    def __len__(self):
        return len(self.sequences)

class RandomSequence(data.Dataset):
    def __init__(self, config):
        self.config = config
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA = {aa:i for i, aa in enumerate(AA)}
        self.AA['pad'] = len(self.AA)
        self.idx_to_AA = {i:aa for aa, i in self.AA.items()}
        self.vocab_size = len(self.AA)
        self.true_seq = [line.strip().split(' ') for line in open(f"{self.config['path']}/{self.config['target']}{self.config['filename'][:-4]}.txt")][0]
        self.data = self.generate()

    def generate(self):
        seq_to_idx = []
        i = 0
        while 1:
            sq_len = np.random.choice(self.config['seq_len'], 1)[0]
            if sq_len == 0:
                continue
            idx = np.random.choice(len(self.AA)-1, sq_len)
            seq = ''.join(self.idx_to_AA[j] for j in idx)
            if seq in self.true_seq:
                continue
            idx = np.concatenate([idx, np.array([self.AA['pad']] * (self.config['seq_len']-len(idx)))])
            seq_to_idx.append(idx)
            i += 1
            if i == self.config['nm_gen_seq']:
                break
        return seq_to_idx

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

    def __len__(self):
        return len(self.data)