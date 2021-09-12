from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from weighted_retraining.weighted_retraining.utils import print_flush
from weighted_retraining.weighted_retraining.topology.topology_model import TopologyVAE
from weighted_retraining.weighted_retraining.topology.topology_dataset import score_function, get_topology_target_path


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

def get_latent_encodings(use_test_set, use_full_data_for_gp, model, data_file, data_scores, data_imgs,
                         n_best, n_rand, tkwargs: Dict[str, Any],
                         bs=5000, bs_true_eval: int = 256, repeat: int = 10, return_inds: bool = False):
    """ get latent encodings and split data into train and test data """

    print_flush("\tComputing latent training data encodings and corresponding scores...")
    n_batches = int(np.ceil(len(data_imgs) / bs))

    if n_best > 0 and n_rand > 0 and (n_best + n_rand) < len(data_scores):
        # do not encode all data, it's too long, only encode the number of points needed (w.r.t. n_best+n_rand)
        sorted_idx = np.argsort(-data_scores)
        best_idx = sorted_idx[:n_best]
        rand_idx = sorted_idx[np.random.choice(list(range(n_best + 1, len(data_scores))), n_rand, replace=False)]
        n_best_scores = data_scores[best_idx]
        n_best_data = data_imgs[best_idx]
        n_rand_scores = data_scores[rand_idx]
        n_rand_data = data_imgs[rand_idx]
        # concatenate and then shuffle
        scores_best_cat_rand = np.concatenate([n_best_scores, n_rand_scores])
        data_best_cat_rand = np.concatenate([n_best_data, n_rand_data])
        cat_idx = np.arange(len(scores_best_cat_rand))
        cat_shuffled_idx = np.random.choice(cat_idx, len(cat_idx))
        scores_best_cat_rand = scores_best_cat_rand[cat_shuffled_idx]
        data_best_cat_rand = data_best_cat_rand[cat_shuffled_idx]
        n_batches = int(np.ceil(len(data_best_cat_rand) / bs))
        Xs = [model.encode_to_params(
            torch.from_numpy(data_best_cat_rand[i * bs:(i + 1) * bs]).to(**tkwargs).unsqueeze(1)
        )[0].detach().cpu().numpy() for i in tqdm(range(n_batches))]
    else:
        Xs = [model.encode_to_params(
            torch.from_numpy(data_imgs[i * bs:(i + 1) * bs]).to(**tkwargs).unsqueeze(1)
        )[0].detach().cpu().numpy() for i in tqdm(range(n_batches))]
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


def evaluate_enc(zs: Tensor, model: TopologyVAE, tkwargs, bs=256) -> np.ndarray:
    model.eval()
    model.vae.to(**tkwargs)
    target = np.load(get_topology_target_path())
    dl = DataLoader(zs, batch_size=bs)
    vals = []
    with torch.no_grad():
        for z in tqdm(dl):
            z.to(**tkwargs)
            vals = np.append(vals, score_function(model.decode_deterministic(z), target))
    return vals


def get_rec_x_error(model: TopologyVAE, tkwargs, xs: Tensor, zs: Tensor, bs=256) -> Tensor:
    """
    Get reconstruction errors between x and decoder(encoder(x))

    Args:
        model: equation auto-encoder model
        tkwargs: kwargs for dtype and device
        xs: x original space
        zs: latent embeddings for the x
        bs: batch size

    Returns:
        errors: reconstruction error for each input

    """
    model.eval()
    model.to(**tkwargs)
    dataset = TensorDataset(zs, xs)
    dl = DataLoader(dataset, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z, x in tqdm(dl):
            z = z.to(**tkwargs)
            x = x.to(**tkwargs)
            error = model.decoder_loss(z, x, return_batch=True).mean((-1, -2))
            assert error.shape == (len(z), 1), (error.shape, z.shape, x.shape)
            if errors is None:
                errors = error
            else:
                errors = torch.vstack([errors, error])
    return errors


def get_rec_z_error(model: TopologyVAE, tkwargs, zs: Tensor, bs=256, *args, **kwargs) -> Tensor:
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
    model.to(**tkwargs)
    dl = DataLoader(zs, batch_size=bs)

    errors: Optional[Tensor] = None
    with torch.no_grad():
        for z in tqdm(dl):
            z = z.to(**tkwargs)
            new_imgs = model.decode_deterministic(z=z)
            z_rec = model.encode_to_params(new_imgs)[0]
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
