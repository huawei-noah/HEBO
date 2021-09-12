""" Code for chem datasets """

import pickle
from typing import Any, Optional, Dict, List, Iterable

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.auto import tqdm

import weighted_retraining.weighted_retraining.chem.jtnn.datautils as datautils
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.chem_utils import standardize_smiles, penalized_logP, QED_score
from weighted_retraining.weighted_retraining.chem.jtnn import MolTreeFolder, MolTreeDataset, Vocab, MolTree
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import TargetMolTreeDataset
from weighted_retraining.weighted_retraining.utils import print_flush

NUM_WORKERS = 4


##################################################
# Data preprocessing code
##################################################
def get_vocab_from_tree(tree: MolTree):
    cset = set()
    for c in tree.nodes:
        cset.add(c.smiles)
    return cset


def get_vocab_from_smiles(smiles):
    """ Get the set of all vocab items for a given smiles """
    mol = MolTree(smiles)
    return get_vocab_from_tree(mol)


def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


##################################################
# Pytorch lightning data classes
##################################################
class WeightedMolTreeFolder(MolTreeFolder):
    """ special weighted mol tree folder """

    def __init__(self, prop, property_dict, data_weighter, n_init_points: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.predict_target = kwargs.get('predict_target')
        self.metric_loss = kwargs.get('metric_loss')
        self.n_init_points = n_init_points
        # Store all the underlying data
        self.data = []
        for idx in range(len(self.data_files)):
            data = self._load_data_file(idx)
            self.data += data
            del data

        self.data_weighter = data_weighter

        if n_init_points is not None:
            indices = np.random.randint(0, len(self.data), n_init_points)
            self.data = [self.data[index] for index in indices]

        self.property = prop
        if self.property == "logP":
            self.prop_func = penalized_logP
        elif self.property == "QED":
            self.prop_func = QED_score
        else:
            raise NotImplementedError(self.property)
        self._set_data_properties(property_dict)

    def dataset_target_preprocess(self, targets: np.ndarray) -> Optional[np.ndarray]:
        """ Depending on the configuration, Dataloader should provide (normalized) targets """
        if self.predict_target:
            return targets
        if self.metric_loss is not None:
            # when using both metric learning and target prediction, min-max normalization is done when the
            # loss is computed
            assert not self.predict_target
            if self.metric_loss in ('contrastive', 'triplet', 'log_ratio'):
                m, M = targets.min(), targets.max()
                return (targets - m) / (M - m)
            else:
                raise ValueError(f'{self.metric_loss} not supported')
        return None

    def _set_data_properties(self, property_dict):
        """ Set various properties from the dataset """

        # Extract smiles from the data
        self.smiles = [t.smiles for t in self.data]
        self.canonic_smiles = list(map(standardize_smiles, self.smiles))

        # Set the data length
        self._len = len(self.data) // self.batch_size

        # Calculate any missing properties
        if not set(self.canonic_smiles).issubset(set(property_dict.keys())):
            for s in tqdm(
                    set(self.canonic_smiles) - set(property_dict), desc="calc properties"
            ):
                property_dict[s] = self.prop_func(s)

        # Randomly check that the properties match the ones calculated
        # Check first few, random few, then last few
        max_check_size = min(10, len(self.data))
        prop_check_idxs = list(
            np.random.choice(
                len(self.canonic_smiles), size=max_check_size, replace=False
            )
        )
        prop_check_idxs += list(range(max_check_size)) + list(range(-max_check_size, 0))
        prop_check_idxs = sorted(list(set(prop_check_idxs)))
        for i in prop_check_idxs:
            s = self.canonic_smiles[i]
            assert np.isclose(
                self.prop_func(s), property_dict[s], rtol=1e-3, atol=1e-4
            ), f"score for smiles {s} doesn't match property dict for property {self.property}"

        # Finally, set properties attribute!
        self.data_properties = np.array([property_dict[s] for s in self.canonic_smiles])
        self.vae_data_properties = self.dataset_target_preprocess(self.data_properties)
        self.m, self.M = self.data_properties.min(), self.data_properties.max()

        # Calculate weights (via maximization)
        self.weights = self.data_weighter.weighting_function(self.data_properties)

        # Sanity check
        assert len(self.data) == len(self.data_properties) == len(self.weights)

    def __len__(self):
        return self._len

    def __iter__(self):
        """ iterate over the dataset with weighted choice """

        # Shuffle the data in a weighted way
        weighted_idx_shuffle = np.random.choice(
            len(self.weights),
            size=len(self.weights),
            replace=True,
            p=self.weights / self.weights.sum(),
        )

        # Make batches
        shuffled_data = [self.data[i] for i in weighted_idx_shuffle]
        batches = [
            shuffled_data[i: i + self.batch_size]
            for i in range(0, len(shuffled_data), self.batch_size)
        ]
        if len(batches[-1]) < self.batch_size:
            batches.pop()

        if self.predict_target or self.metric_loss is not None:
            suffled_property_data = self.vae_data_properties[weighted_idx_shuffle]
            property_batches = [
                suffled_property_data[i: i + self.batch_size]
                for i in range(0, len(suffled_property_data), self.batch_size)
            ]
            if len(property_batches[-1]) < self.batch_size:
                property_batches.pop()
            dataset = TargetMolTreeDataset(batches, self.vocab, property_batches, self.assm)
        else:
            dataset = MolTreeDataset(batches, self.vocab, self.assm)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
        )
        for b in dataloader:
            yield b
        del dataset, dataloader, shuffled_data


class WeightedJTNNDataset(pl.LightningDataModule):
    """ dataset with property weights. Needs to load all data into memory """

    def __init__(self, hparams, data_weighter):
        super().__init__()
        self.train_path = hparams.train_path
        self.val_path = hparams.val_path
        self.vocab_file = hparams.vocab_file
        self.batch_size = hparams.batch_size
        self.property = hparams.property
        self.property_file = hparams.property_file
        self.data_weighter = data_weighter
        self.predict_target: bool = hparams.predict_target
        self.maximize: bool = True
        self.metric_loss = hparams.metric_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--train_path", required=True)
        data_group.add_argument("--val_path", required=False, default=None)
        data_group.add_argument("--vocab_file", required=True)
        data_group.add_argument("--batch_size", type=int, default=32)
        data_group.add_argument(
            "--property", type=str, choices=["logP", "QED"], default="logP"
        )
        data_group.add_argument(
            "--property_file",
            type=str,
            default=None,
            help="dictionary file mapping smiles to properties. Optional but recommended",
        )
        return parent_parser

    def setup(self, stage=None, n_init_points: Optional[bool] = None):

        # Create vocab
        with open(self.vocab_file) as f:
            self.vocab = Vocab([x.strip() for x in f.readlines()])

        # Read in properties
        if self.property_file is None:
            property_dict = dict()
        else:
            with open(self.property_file, "rb") as f:
                property_dict = pickle.load(f)

        self.train_dataset = WeightedMolTreeFolder(
            self.property,
            property_dict,
            self.data_weighter,
            n_init_points,
            self.train_path,
            self.vocab,
            self.batch_size,
            num_workers=NUM_WORKERS,
            predict_target=self.predict_target,
            metric_loss=self.metric_loss,
        )

        # Val dataset, if given
        if self.val_path is None:
            self.val_dataset = None
        else:
            self.val_dataset = WeightedMolTreeFolder(
                self.property,
                property_dict,
                self.data_weighter,
                None,
                self.val_path,
                self.vocab,
                self.batch_size,
                num_workers=NUM_WORKERS,
                predict_target=self.predict_target,
                metric_loss=self.metric_loss
            )

        self.specific_setup()

    def specific_setup(self):
        if self.predict_target and self.metric_loss is not None:
            # Set target min-max normalisation constants
            self.training_m, self.training_M = self.train_dataset.m, self.train_dataset.M
            self.validation_m, self.validation_M = self.val_dataset.m, self.val_dataset.M

    def append_train_data(self, smiles_new, z_prop, append_all_data: bool = None):
        dset = self.train_dataset

        assert len(smiles_new) == len(z_prop)

        # Check which smiles need to be added!
        can_smiles_set = set(dset.canonic_smiles)
        prop_dict = {s: p for s, p in zip(dset.canonic_smiles, dset.data_properties)}

        # Total vocabulary set
        vocab_set = set(self.vocab.vocab)

        # Go through and do the addition
        s_add = []
        data_to_add = []
        props_to_add = []
        for s, prop in zip(smiles_new, z_prop):
            if s is None:
                continue
            s_std = standardize_smiles(s)
            if s_std not in can_smiles_set:  # only add new smiles

                # tensorize data
                tree_tensor = tensorize(s_std)

                # Make sure satisfies vocab check
                v_set = get_vocab_from_tree(tree_tensor)
                if v_set <= vocab_set:
                    # Add to appropriate trackers
                    can_smiles_set.add(s_std)
                    s_add.append(s_std)
                    data_to_add.append(tree_tensor)
                    props_to_add.append(prop)

                    # Update property dict for later
                    prop_dict[s_std] = prop
        props_to_add = np.array(props_to_add)

        # Either add or replace the data, depending on the mode
        if self.data_weighter.weight_type == "fb" and not append_all_data:
            # Find top quantile
            cutoff = np.quantile(props_to_add, self.data_weighter.weight_quantile)
            indices_to_add = props_to_add >= cutoff

            # Filter all but top quantile
            s_add = [s_add[i] for i, ok in enumerate(indices_to_add) if ok]
            data_to_add = [data_to_add[i] for i, ok in enumerate(indices_to_add) if ok]
            props_to_add = props_to_add[indices_to_add]
            assert len(s_add) == len(data_to_add) == len(props_to_add)

            # Replace the first few data points
            dset.data = dset.data[len(data_to_add):] + data_to_add

        else:
            dset.data += data_to_add

        # Now recalcuate weights/etc
        dset._set_data_properties(prop_dict)
        self.specific_setup()
        # Return what was successfully added to the dataset
        return s_add, props_to_add

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=lambda x: [x], batch_size=None)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset, collate_fn=lambda x: [x], batch_size=None
            )


def split_dataset(X, y, split=0.9, return_inds: bool = False):
    """ split the data into a train and test set """

    n = X.shape[0]
    permutation = np.random.choice(n, n, replace=False)
    train_inds = permutation[:np.int(np.round(split * n))]
    test_inds = permutation[np.int(np.round(split * n)):]
    X_train = X[train_inds]
    y_train = y[train_inds]

    X_test = X[test_inds]
    y_test = y[test_inds]
    if return_inds:
        return X_train, y_train, train_inds, X_test, y_test, test_inds
    else:
        return X_train, y_train, X_test, y_test


def subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand, return_inds: bool = False):
    """ subsample dataset for training the sparse GP """

    if use_test_set:
        res_split = split_dataset(X, y, return_inds=return_inds)
        if return_inds:
            X_train, y_train, train_inds, X_test, y_test, test_inds = res_split
        else:
            X_train, y_train, X_test, y_test = res_split
    else:
        X_train, y_train, X_test, y_test = X, y, None, None
        train_inds = np.arange(len(X))
        test_inds = np.arange(0)

    if len(y_train) < n_best + n_rand:
        n_best = int(n_best / (n_best + n_rand) * len(y_train))
        n_rand = int(n_rand / (n_best + n_rand) * len(y_train))

    if not use_full_data_for_gp:
        # pick n_best best points and n_rand random points
        best_idx = np.argsort(np.ravel(y_train))[:n_best]
        rand_idx = np.argsort(np.ravel(y_train))[np.random.choice(
            list(range(n_best, len(y_train))), n_rand, replace=False)]
        all_idx = np.concatenate([best_idx, rand_idx])
        X_train = X_train[all_idx, :]
        y_train = y_train[all_idx]
        train_inds = train_inds[all_idx]
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    save_data(X_train, y_train, X_test, y_test, X_mean, X_std, y_mean, y_std, data_file)
    if return_inds:
        return X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std, train_inds, test_inds
    else:
        return X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std


def save_data(X_train, y_train, X_test, y_test, X_mean, X_std, y_mean, y_std, data_file):
    """ save data """

    X_train_ = (X_train - X_mean) / X_std
    y_train_ = (y_train - y_mean) / y_std
    np.savez_compressed(
        data_file,
        X_train=np.float32(X_train_),
        y_train=np.float32(y_train_),
        X_test=np.float32(X_test),
        y_test=np.float32(y_test),
    )


def _choose_best_rand_points(n_best_points: int, n_rand_points: int, dataset: WeightedMolTreeFolder):
    chosen_point_set = set()

    # Best scores at start
    targets_argsort = np.argsort(-dataset.data_properties.flatten())
    for i in range(n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=n_rand_points + n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (
                n_rand_points + n_best_points
        ):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (n_rand_points + n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_mol_trees(model, mol_trees, batch_size: int = 64):
    mu_list = []
    with torch.no_grad():
        for i in trange(
                0, len(mol_trees), batch_size, desc="encoding GP points", leave=False
        ):
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = datautils.tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    return mu


def get_latent_encodings(use_test_set, use_full_data_for_gp, model, data_file,
                         data_set: WeightedMolTreeFolder,
                         n_best, n_rand, true_vals: bool, tkwargs: Dict[str, Any],
                         bs=64, return_inds: bool = False):
    """ get latent encodings and split data into train and test data """

    print_flush("\tComputing latent training data encodings and corresponding scores...")

    if len(data_set) < n_best + n_rand:
        n_best, n_rand = int(n_best / (n_best + n_rand) * len(data_set)), int(
            n_rand / (n_best + n_rand) * len(data_set))
        n_rand += 1 if n_best + n_rand < len(data_set) else 0

    if use_full_data_for_gp:
        chosen_indices = np.arange(len(data_set))
    else:
        chosen_indices = _choose_best_rand_points(n_best, n_rand, data_set)
    mol_trees = [data_set.data[i] for i in chosen_indices]
    targets = data_set.data_properties[chosen_indices]

    # Next, encode these mol trees
    latent_points = _encode_mol_trees(model, mol_trees, batch_size=bs)

    targets = targets.reshape((-1, 1))

    # problem with train_inds returned by ubsample_dataset is they are train indices within passed points and not
    # indices of the original dataset
    if not use_full_data_for_gp:
        assert not use_test_set
        X_mean, X_std = latent_points.mean(), latent_points.std()
        y_mean, y_std = targets.mean(), targets.std()
        save_data(latent_points, targets, None, None, X_mean, X_std, y_mean, y_std, data_file)
        if return_inds:
            return latent_points, targets, None, None, X_mean, y_mean, X_std, y_std, chosen_indices, None
        else:
            return latent_points, targets, None, None, X_mean, y_mean, X_std, y_std

    return subsample_dataset(latent_points, targets, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand,
                             return_inds=return_inds)


def get_rec_x_error(model: JTVAE, tkwargs, data: List) -> Iterable[Tensor]:
    """
    Get reconstruction errors between x and decoder(encoder(x))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        data: smile encodings of molecules
        zs: latent embeddings for the molecules
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.to(**tkwargs)
    errors: Optional[Tensor] = None
    success_ind = []
    with torch.no_grad():
        for i in trange(len(data)):
            batch_x = ([data[i]],)
            try:
                batch_z = model.encode_to_params(datautils.tensorize(batch_x[0], model.jtnn_vae.vocab, assm=True))[0]
                error = model.decoder_loss(batch_z, batch_x).view(-1, 1)
                assert error.shape == (len(batch_z), 1), (error.shape, batch_z.shape, len(batch_x[0]))
                if errors is None:
                    errors = error
                else:
                    errors = torch.vstack([errors, error])
                success_ind.append(i)
            except:
                pass
    return errors, success_ind


def get_rec_z_error(model: JTVAE, tkwargs, zs: Tensor, bs=256, n_decode_attempts=100) -> Tensor:
    """
    Get reconstruction errors between z and encoder(decoder(z))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        zs: latent embeddings for the `one_hots`
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.to(**tkwargs)
    dl = DataLoader(zs, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z in tqdm(dl):
            z = z.to(**tkwargs)
            new_chems = model.decode_deterministic(z=z)
            z_rec = model.encode_to_params(new_chems)[0]
            error = torch.norm(z - z_rec, dim=1, keepdim=True).pow(2)
            assert error.shape == (len(z), 1), (error.shape, z.shape, z_rec.shape)
            if errors is None:
                errors = error
            else:
                errors = torch.vstack([errors, error])
    return errors


def append_trainset_torch(X_train: Tensor, y_train: Tensor, new_inputs: Tensor, new_scores: Tensor,
                          y_errors: Tensor = None, new_errors: Tensor = None):
    """ add new inputs and scores to training set """

    assert new_scores.ndim == 2 and new_scores.shape[-1] == 1, new_scores.shape
    if len(new_inputs) > 0:
        X_train = torch.cat([X_train, new_inputs.to(X_train)], 0)
        y_train = torch.cat([y_train, new_scores.to(y_train)], 0)
        if new_errors is not None:
            y_errors = torch.cat([y_errors, new_errors.to(y_errors)], 0)
    if new_errors is not None:
        return X_train, y_train, y_errors
    return X_train, y_train


def append_trainset_torch(X_train: Tensor, y_train: Tensor, new_inputs: Tensor, new_scores: Tensor,
                          y_errors: Tensor = None, new_errors: Tensor = None):
    """ add new inputs and scores to training set """

    assert new_scores.ndim == 2 and new_scores.shape[-1] == 1, new_scores.shape
    if len(new_inputs) > 0:
        X_train = torch.cat([X_train, new_inputs.to(X_train)], 0)
        y_train = torch.cat([y_train, new_scores.to(y_train)], 0)
        if new_errors is not None:
            y_errors = torch.cat([y_errors, new_errors.to(y_errors)], 0)
    if new_errors is not None:
        return X_train, y_train, y_errors
    return X_train, y_train
