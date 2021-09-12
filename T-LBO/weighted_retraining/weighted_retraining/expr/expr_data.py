""" Code for loading and manipulating the arithmetic expression data """

import os
from typing import Sequence, Any, Dict, Optional, Iterable, Collection

import h5py
import numpy as np
import torch
# noinspection PyUnresolvedReferences
from numpy import exp, sin
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from weighted_retraining.weighted_retraining.expr.equation_vae import EquationGrammarModelTorch
from weighted_retraining.weighted_retraining.utils import print_flush


def load_data_str(data_dir):
    """ load the arithmetic expression data in string format """
    fname = 'equation2_15_dataset.txt'
    with open(data_dir / fname) as f:
        eqs = f.readlines()
    for i in range(len(eqs)):
        eqs[i] = eqs[i].strip().replace(' ', '')
    return eqs


def load_data_enc(data_dir):
    """ load the arithmetic expression dataset in one-hot encoded format """

    fname = 'eq2_grammar_dataset.h5'
    h5f = h5py.File(data_dir / fname, 'r')
    data = h5f['data'][:]
    h5f.close()

    return data


def get_initial_dataset_and_weights(data_dir, ignore_percentile, n_data):
    """ get the initial dataset (with corresponding scores) and the sample weights """

    # load equation dataset, both one-hot encoded and as plain strings, and compute corresponding scores
    print(os.getcwd())
    data_str = load_data_str(data_dir)
    data_enc = load_data_enc(data_dir)
    data_scores = score_function(data_str)

    # subsample data based on the desired percentile and # of datapoints
    perc = np.percentile(data_scores, ignore_percentile)
    perc_idx = data_scores >= perc
    data_idx = np.random.choice(sum(perc_idx), min(n_data, sum(perc_idx)), replace=False)
    data_str = list(np.array(data_str)[perc_idx][data_idx])
    data_enc = data_enc[perc_idx][data_idx]
    data_scores = data_scores[perc_idx][data_idx]

    return data_str, data_enc, data_scores


def update_dataset_and_weights(new_inputs, new_scores, data_str, data_enc, data_scores, model):
    """ update the dataet and the sample weights """

    # discard invalid (None) inputs and their corresponding scores
    valid_idx = np.array(new_inputs) != None
    valid_inputs = list(new_inputs[valid_idx])
    valid_scores = new_scores[valid_idx]
    print_flush(
        "\tDiscarding {}/{} new inputs that are invalid!".format(len(new_inputs) - len(valid_inputs), len(new_inputs)))

    # add new inputs and scores to dataset, both as plain string and one-hot vector
    print_flush("\tAppending new valid inputs to dataset...")
    data_str += valid_inputs
    new_inputs_one_hot = model.smiles_to_one_hot(valid_inputs)
    data_enc = np.append(data_enc, new_inputs_one_hot, axis=0)
    data_scores = np.append(data_scores, valid_scores)

    return data_str, data_enc, data_scores


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
        n_best, n_rand = int(n_best / (n_best + n_rand) * len(y_train)), int(n_rand / (n_best + n_rand) * len(y_train))
        n_rand += 1 if n_best + n_rand < len(y_train) else 0

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


def append_trainset(X_train, y_train, new_inputs, new_scores):
    """ add new inputs and scores to training set """

    if len(new_inputs) > 0:
        X_train = np.concatenate([X_train, new_inputs], 0)
        y_train = np.concatenate([y_train, new_scores[:, np.newaxis]], 0)
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


def score_function(inputs, target_eq='1 / 3 + x + sin( x * x )', worst=7.0) -> np.ndarray:
    """ compute equation scores of given inputs """

    # define inputs and outputs of ground truth target expression
    x = np.linspace(-10, 10, 1000)
    try:
        yT = np.array(eval(target_eq))
    except NameError as e:
        print(target_eq)
        raise e
    scores = []
    for inp in inputs:
        try:
            y_pred = np.array(eval(inp))
            scores.append(np.minimum(worst, np.log(1 + np.mean((y_pred - yT) ** 2))))
        except:
            scores.append(worst)

    return np.array(scores)


def get_latent_encodings(use_test_set, use_full_data_for_gp, model, data_file, data_scores, data_str,
                         n_best, n_rand, tkwargs: Dict[str, Any],
                         bs=5000, bs_true_eval: int = 256, repeat: int = 10, return_inds: bool = False):
    """ get latent encodings and split data into train and test data """

    print_flush("\tComputing latent training data encodings and corresponding scores...")
    n_batches = int(np.ceil(len(data_str) / bs))

    if n_best > 0 and n_rand > 0 and (n_best + n_rand) < len(data_scores):
        # do not encode all data, it's too long, only encode the number of points needed (w.r.t. n_best+n_rand)
        sorted_idx = np.argsort(data_scores)
        best_idx = sorted_idx[:n_best]
        rand_idx = sorted_idx[np.random.choice(list(range(n_best + 1, len(data_scores))), n_rand, replace=False)]
        n_best_scores = data_scores[best_idx]
        n_best_data = data_str[best_idx]
        n_rand_scores = data_scores[rand_idx]
        n_rand_data = data_str[rand_idx]
        # concatenate and then shuffle
        scores_best_cat_rand = np.concatenate([n_best_scores, n_rand_scores])
        data_best_cat_rand = np.concatenate([n_best_data, n_rand_data])
        cat_idx = np.arange(len(scores_best_cat_rand))
        cat_shuffled_idx = np.random.choice(cat_idx, len(cat_idx))
        scores_best_cat_rand = scores_best_cat_rand[cat_shuffled_idx]
        data_best_cat_rand = data_best_cat_rand[cat_shuffled_idx]
        n_batches = int(np.ceil(len(data_best_cat_rand) / bs))
        Xs = [model.encode(list(data_best_cat_rand[i * bs:(i + 1) * bs])) for i in tqdm(range(n_batches))]
    else:
        Xs = [model.encode(list(data_str[i * bs:(i + 1) * bs])) for i in tqdm(range(n_batches))]
    X = np.concatenate(Xs, axis=0)

    y = scores_best_cat_rand if n_best > 0 and n_rand > 0 and (n_best + n_rand) < len(data_scores) else data_scores
    y = y.reshape((-1, 1))

    if n_best > 0 and n_rand > 0 and (n_best + n_rand) < len(data_scores):
        assert not use_test_set
        assert not use_full_data_for_gp
        X_mean, X_std = X.mean(), X.std()
        y_mean, y_std = y.mean(), y.std()
        save_data(X, y, None, None, X_mean, X_std, y_mean, y_std, data_file)
        if return_inds:
            train_inds = np.concatenate([best_idx, rand_idx])[cat_shuffled_idx]
            return X, y, None, None, X_mean, y_mean, X_std, y_std, train_inds, None
        else:
            return X, y, None, None, X_mean, y_mean, X_std, y_std
    return subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand,
                             return_inds=return_inds)


def get_latent_encodings_aux(model: EquationGrammarModelTorch, data_str: Sequence[str], bs: int = 5000) -> np.ndarray:
    """ get latent encodings of inputs in given dataset 
    
    Args:
        bs: batch size (model will encode inputs by batches of size `bs`)
        model: generative model having `encode` method
        data_str: sequence of inputs to encode
    """
    n_batches = int(np.ceil(len(data_str) / bs))
    Xs = [model.encode(list(data_str[i * bs:(i + 1) * bs])) for i in tqdm(range(n_batches))]
    return np.concatenate(Xs, axis=0)


def evaluate_enc(zs: Tensor, model: EquationGrammarModelTorch, tkwargs, bs=256, repeat=10) -> np.ndarray:
    model.eval()
    model.vae.to(**tkwargs)

    dl = DataLoader(zs, batch_size=bs)
    vals = []
    with torch.no_grad():
        for z in tqdm(dl):
            z.to(**tkwargs)
            vals = np.append(vals, np.vstack(
                [score_function(model.decode_from_latent_space(z, n_decode_attempts=5)) for _ in range(repeat)]).mean(
                0))
    return vals


def get_rec_x_error(model: EquationGrammarModelTorch, tkwargs, one_hots: Tensor, zs: Tensor,
                    bs=256) -> Tensor:
    """
    Get reconstruction errors between x and decoder(encoder(x))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        one_hots: one-hot encoding of equations
        zs: latent embeddings for the `one_hots`
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.vae.to(**tkwargs)
    dataset = TensorDataset(zs, one_hots)
    dl = DataLoader(dataset, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z, one_hot in tqdm(dl):
            z = z.to(**tkwargs)
            one_hot = one_hot.to(**tkwargs)
            error = model.vae.decoder_loss(z, one_hot, reduction='none').mean((-1, -2)).unsqueeze(1)
            assert error.shape == (len(z), 1), (error.shape, z.shape, one_hot.shape)
            if errors is None:
                errors = error
            else:
                errors = torch.vstack([errors, error])
    return errors


def get_rec_z_error(model: EquationGrammarModelTorch, tkwargs, zs: Tensor, bs=256, n_decode_attempts=100) -> Tensor:
    """
    Get reconstruction errors between z and encoder(decoder(x))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        zs: latent embeddings for the `one_hots`
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.vae.to(**tkwargs)
    dl = DataLoader(zs, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z in tqdm(dl):
            z = z.to(**tkwargs)
            new_exprs = model.decode_from_latent_space(zs=z, n_decode_attempts=n_decode_attempts)
            one_hots = torch.from_numpy(model.smiles_to_one_hot(list(new_exprs))).to(**tkwargs)
            z_rec = model.vae.encode_to_params(one_hots)[0]
            error = torch.norm(z - z_rec, dim=1, keepdim=True).pow(2)
            assert error.shape == (len(z), 1), (error.shape, z.shape, z_rec.shape)
            if errors is None:
                errors = error
            else:
                errors = torch.vstack([errors, error])
    return errors


def get_rec_z_error_safe(model: EquationGrammarModelTorch, tkwargs, zs: Tensor, bs=256,
                         n_decode_attempts=100) -> Iterable[Tensor]:
    """
    Get reconstruction errors between z and encoder(decoder(x))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        zs: latent embeddings for the `one_hots`
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.vae.to(**tkwargs)
    dl = DataLoader(zs, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z in tqdm(dl):
            z = z.to(**tkwargs)
            new_exprs = model.decode_from_latent_space(zs=z, n_decode_attempts=n_decode_attempts)
            valid_idx = np.array(new_exprs) != None
            invalid_idx = np.array(new_exprs) == None
            new_exprs_valid = list(new_exprs[valid_idx])
            one_hots = torch.from_numpy(model.smiles_to_one_hot(list(new_exprs_valid))).to(**tkwargs)
            z_rec = model.vae.encode_to_params(one_hots)[0]
            error = torch.norm(z[valid_idx] - z_rec, dim=1, keepdim=True).pow(2)
            assert error.shape == (len(z), 1), (error.shape, z.shape, z_rec.shape)
            if errors is None:
                errors = error
                invalid_idx_list = [invalid_idx]
            else:
                errors = torch.vstack([errors, error])
                invalid_idx_list.append(invalid_idx)
    return errors, invalid_idx_list


def get_rec_error_emb(model: EquationGrammarModelTorch, tkwargs, exprs: Collection[str], bs=256):
    model.eval()
    model.vae.to(**tkwargs)
    n_batches = int(np.ceil(len(exprs) / bs))
    one_hots = torch.from_numpy(
        np.concatenate(
            [model.smiles_to_one_hot(list(exprs[i * bs:(i + 1) * bs])) for i in tqdm(range(n_batches))])
    ).to(**tkwargs)

    dataloader = DataLoader(one_hots, batch_size=bs)
    zs = None
    with torch.no_grad():
        for one_hot in tqdm(dataloader):
            z = model.vae.encode_to_params(one_hot)[0]
            if zs is None:
                zs = z
            else:
                zs = torch.vstack([zs, z])
    return get_rec_x_error(model, tkwargs=tkwargs, one_hots=one_hots, zs=zs, bs=bs)
