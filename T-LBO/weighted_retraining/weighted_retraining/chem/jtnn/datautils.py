import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from weighted_retraining.weighted_retraining.chem.jtnn.mol_tree import MolTree
import numpy as np
from weighted_retraining.weighted_retraining.chem.jtnn.jtnn_enc import JTNNEncoder
from weighted_retraining.weighted_retraining.chem.jtnn.mpn import MPN
from weighted_retraining.weighted_retraining.chem.jtnn.jtmpn import JTMPN
import pickle
import os, random


class PairTreeFolder(object):
    def __init__(
            self,
            data_folder,
            vocab,
            batch_size,
            num_workers=4,
            shuffle=True,
            y_assm=True,
            replicate=None,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [
                data[i: i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda x: x[0],
            )

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeFolder(IterableDataset):
    def __init__(
            self,
            data_folder,
            vocab,
            batch_size,
            num_workers=8,
            shuffle=True,
            assm=True,
            replicate=None,
            **kwargs,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder) if fn.endswith(".pkl")]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def _get_dataset_dataloader(
            self, data, shuffle_override=None, drop_last_batch=True
    ):
        """
        Method to separate reading out data for fast jtnn, so that the moltrees
        can be intercepted and used for other things (e.g. reconstruction)
        """

        # Decide whether to shuffle
        if shuffle_override is None:
            shuffle = self.shuffle
        else:
            shuffle = shuffle_override

        # Potentially shuffle the data
        if shuffle:
            random.shuffle(data)  # shuffle data before batch

        # Make batches
        batches = [
            data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)
        ]
        if drop_last_batch and len(batches[-1]) < self.batch_size:
            batches.pop()

        dataset = MolTreeDataset(batches, self.vocab, self.assm)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
        )
        return dataset, dataloader

    def _load_data_file(self, idx):
        """ Load data file # idx """
        fn = self.data_files[idx]
        fn = os.path.join(self.data_folder, fn)
        print(f"loading {fn}")
        with open(fn, "rb") as f:
            data = pickle.load(f)
        return data

    def __iter__(self):
        for idx in range(len(self.data_files)):
            data = self._load_data_file(idx)

            # Get moltree dataset and corresponding dataloader
            dataset, dataloader = self._get_dataset_dataloader(data)

            for b in dataloader:
                yield b

            del data, dataset, dataloader


class PairTreeDataset(Dataset):
    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return (
            tensorize(batch0, self.vocab, assm=False),
            tensorize(batch1, self.vocab, assm=self.y_assm),
        )


class MolTreeDataset(Dataset):
    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)


class TargetMolTreeDataset(Dataset):
    def __init__(self, data, vocab, target, assm=True):
        self.data = data
        self.vocab = vocab
        self.target = target
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, target=self.target[idx], assm=self.assm)


def tensorize(tree_batch, vocab, assm=True, target=None):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False and target is None:
        return tree_batch, jtenc_holder, mpn_holder
    elif assm is False and target is not None:
        target_tensor = torch.FloatTensor(target)
        return tree_batch, jtenc_holder, mpn_holder, target_tensor

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    if target is not None:
        target_tensor = torch.FloatTensor(target)
        return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx), target_tensor
    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
