import os
from typing import Optional

import torch
from torch import Tensor

from utils.utils_save import get_data_root
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def get_topology_dataset_path():
    return os.path.join(get_data_root(), f'topology_data/topology_data.npz')


def get_topology_target_path():
    return os.path.join(get_data_root(), f'topology_data/target.npy')


def get_topology_start_score_path():
    return os.path.join(get_data_root(), f'topology_data/start_score.npy')


def get_topology_binary_dataset_path():
    return os.path.join(get_data_root(), f'topology_data/topology_data_bin.npz')


def get_topology_binary_target_path():
    return os.path.join(get_data_root(), f'topology_data/target_bin.npy')


def get_topology_binary_start_score_path():
    return os.path.join(get_data_root(), f'topology_data/start_score_bin.npy')


def score_function(predicted_image, target_image, metric: Optional[str] = 'cos'):
    if isinstance(predicted_image, np.ndarray):
        predicted_image = torch.from_numpy(predicted_image).to(torch.float)
    if isinstance(predicted_image, Tensor):
        predicted_image = predicted_image.detach().cpu().to(torch.float)
    if isinstance(target_image, np.ndarray):
        target_image = torch.from_numpy(target_image).to(torch.float)
    if isinstance(target_image, Tensor):
        target_image = target_image.detach().cpu().to(torch.float)

    if metric == 'cos':
        pred_flat = predicted_image.view(*predicted_image.shape[:-2], -1)
        target_flat = target_image.flatten()
        if target_flat.ndim < pred_flat.ndim:
            target_flat = target_flat.unsqueeze(0)
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        return cosine_sim(pred_flat, target_flat).cpu().numpy()
    elif metric == 'jaccard':
        return sp.spatial.distance.jaccard(predicted_image.flatten(), target_image.flatten())
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")


def gen_dataset_from_all_files(data_root):
    all_data = []
    for i in range(10000):
        with np.load(os.path.join(data_root, f"topology_data/TOP4040/{i}.npz")) as npz:
            all_data.append(npz["arr_0"][-1].reshape(1, 40, 40))
    all_data = np.concatenate(all_data)
    nprange = np.arange(len(all_data))
    all_data = all_data[np.random.choice(nprange, len(all_data))]  # shuffle data
    target_idx = 3000
    target = all_data[target_idx]
    all_data = all_data[np.delete(nprange, target_idx)]  # remove target from all_data
    all_scores = score_function(all_data, target)
    start_score = all_scores.max()
    np.savez_compressed(os.path.join(data_root, "topology_data/topology_data"), data=all_data, scores=all_scores)
    np.save(os.path.join(data_root, "topology_data/target"), target)
    np.save(os.path.join(data_root, "topology_data/start_score"), start_score)
    plt.imshow(target)
    plt.savefig(os.path.join(data_root, "topology_data/target.pdf"))
    plt.close()


def gen_binary_dataset_from_all_files(data_root):
    all_data = []
    for i in range(10000):
        with np.load(os.path.join(data_root, f"topology_data/TOP4040/{i}.npz")) as npz:
            continuous = npz["arr_0"][-1].reshape(1, 40, 40)
            binarized = np.where(continuous > 0.5, 1, 0)
            all_data.append(binarized)
    all_data = np.concatenate(all_data)
    nprange = np.arange(len(all_data))
    all_data = all_data[np.random.choice(nprange, len(all_data))]  # shuffle data
    target_idx = 3000
    target = all_data[target_idx]
    all_data = all_data[np.delete(nprange, target_idx)]  # remove target from all_data
    all_scores = score_function(all_data, target)
    start_score = all_scores.max()
    np.savez_compressed(os.path.join(data_root, "topology_data/topology_data_bin"), data=all_data, scores=all_scores)
    np.save(os.path.join(data_root, "topology_data/target_bin"), target)
    np.save(os.path.join(data_root, "topology_data/start_score_bin"), start_score)
    plt.imshow(target)
    plt.savefig(os.path.join(data_root, "topology_data/target_bin.pdf"))
    plt.close()

