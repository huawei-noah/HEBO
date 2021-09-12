from typing import Dict, Any

import torch
from tqdm import tqdm

from weighted_retraining.weighted_retraining.shapes.shapes_model import ShapesVAE

from weighted_retraining.weighted_retraining.utils import print_flush
import numpy as np


def get_latent_encodings(use_test_set: bool, use_full_data_for_gp: bool, model: ShapesVAE, data_file: str,
                         data_imgs: np.ndarray, data_scores: np.ndarray, n_best: int, n_rand: int,
                         tkwargs: Dict[str, Any],
                         bs=1000):
    """ get latent encodings and split data into train and test data """

    print_flush("\tComputing latent training data encodings and corresponding scores...")
    X = get_latent_encodings_aux(model=model, data_imgs=data_imgs, bs=bs, tkwargs=tkwargs)
    y = data_scores.reshape((-1, 1))

    return _subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand)


def get_latent_encodings_aux(model: ShapesVAE, data_imgs: np.ndarray, tkwargs: Dict[str, Any], bs: int = 1000) -> np.ndarray:
    """ get latent encodings of inputs in given dataset

    Args:
        bs: batch size (model will encode inputs by batches of size `bs`)
        model: generative model having `encode` method
        data_imgs: sequence of inputs to encode
        tkwargs: kwargs for torch dtype and device
    """
    n_batches = int(np.ceil(len(data_imgs) / bs))
    Xs = [model.encode_to_params(torch.from_numpy(data_imgs[i * bs:(i + 1) * bs]).unsqueeze(1).to(**tkwargs))[
              0].cpu().detach().numpy() for i
          in tqdm(range(n_batches))]
    return np.concatenate(Xs, axis=0)


def _subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand):
    """ subsample dataset for training the sparse GP """

    if use_test_set:
        X_train, y_train, X_test, y_test = split_dataset(X, y)
    else:
        X_train, y_train, X_test, y_test = X, y, None, None

    if len(y_train) < n_best + n_rand:
        n_best = int(n_best / (n_best + n_rand) * len(y_train))
        n_rand = int(n_rand / (n_best + n_rand) * len(y_train))

    if not use_full_data_for_gp:
        # pick n_best best points and n_rand random points
        desc_inds = np.argsort(np.ravel(y_train))[::-1]
        best_idx = desc_inds[:n_best]
        rand_idx = desc_inds[np.random.choice(
            list(range(n_best, len(y_train))), n_rand, replace=False)]
        all_idx = np.concatenate([best_idx, rand_idx])
        X_train = X_train[all_idx, :]
        y_train = y_train[all_idx]

    save_data(X_train, y_train, X_test, y_test, data_file)

    return X_train, y_train, X_test, y_test


def save_data(X_train, y_train, X_test, y_test, data_file):
    """ save data """

    np.savez_compressed(
        data_file,
        X_train=np.float32(X_train),
        y_train=np.float32(y_train),
        X_test=np.float32(X_test),
        y_test=np.float32(y_test),
    )


def split_dataset(X, y, split=0.9):
    """ split the data into a train and test set """

    n = X.shape[0]
    permutation = np.random.choice(n, n, replace=False)

    X_train = X[permutation, :][0: np.int(np.round(split * n)), :]
    y_train = y[permutation][0: np.int(np.round(split * n))]

    X_test = X[permutation, :][np.int(np.round(split * n)):, :]
    y_test = y[permutation][np.int(np.round(split * n)):]

    return X_train, y_train, X_test, y_test
