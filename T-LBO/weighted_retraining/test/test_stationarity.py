import os
from typing import Union, Callable, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning import distances
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from weighted_retraining.weighted_retraining.expr.equation_vae import EquationGrammarModelTorch
from weighted_retraining.weighted_retraining.shapes.shapes_model import ShapesVAE
from weighted_retraining.weighted_retraining.topology.topology_model import TopologyVAE


def sample_on_hypersphere(center, radius, n_samples):
    """ return (uniform) samples on the hypersphere centered at provided center and of provided radius """
    # uniformly sample points
    n = torch.distributions.normal.Normal(loc=0, scale=1)
    samples = n.sample((n_samples, *center.size())).to(center)
    # normalise to get points on standard hypersphere (0-center, 1-radius)
    samples.div_(samples.norm(dim=-1, p=2, keepdim=True))
    # adjust samples w.r.t center and radius
    samples.mul_(radius)
    samples.add_(center)
    return samples


def sample_on_hypercube(center, side, n_samples):
    """ return samples on the hypercube of provided side and centered at provided center """
    # uniformly sample points
    u = torch.distributions.uniform.Uniform(0, 1)
    samples = u.sample((n_samples, *center.size())).to(center)
    # randomly pick one dim. of each sample to max to 1 such that it lies on surface (at least a face) of unit hypercube
    d = center.shape[-1]
    col_idx = np.random.choice(np.arange(d), n_samples)
    shuffled_indices = np.random.choice(np.arange(n_samples), n_samples, replace=False)
    row_idx0 = shuffled_indices[:(n_samples // 2)]
    row_idx1 = shuffled_indices[(n_samples // 2):]
    samples[row_idx0, col_idx[:(n_samples // 2)]] = 0
    samples[row_idx1, col_idx[(n_samples // 2):]] = 1
    # adjust for side of hypercube and move to center
    samples.add_(-torch.ones(d).to(center) * 0.5)
    samples.mul_(side)
    samples.add_(center)
    return samples


def sample_on_hyperdiamond(center, side, n_samples):
    """ return (uniform) samples on all quadrants of the simplex of provided side and centered at provided center """
    # sample points following exponential distribution and normalise
    e = torch.distributions.exponential.Exponential(1)
    samples = e.sample((n_samples, *center.size())).to(center)
    samples.div_(samples.sum(dim=-1, keepdim=True))
    # randomly flip some signs to fill all quadrants
    signs = torch.ones_like(samples).to(center)
    rand_mask = np.random.randint(0, 2, size=signs.size())
    signs[rand_mask == 1] = -1
    samples.mul_(signs)
    # adjust for radius and center
    samples.mul_(side)
    samples.add_(center)
    return samples


def average_distance_to_center_target_value(
        center: Tensor,
        radius: float,
        model: Union[EquationGrammarModelTorch, ShapesVAE, TopologyVAE],
        score_function: Callable,
        target: Union[Tensor, np.ndarray],
        n_samples: Optional[int] = 100,
        samples: Optional[Union[Tensor, np.ndarray]] = None,
):
    if samples is not None:
        z_samples = samples
    else:
        z_samples = sample_on_hypersphere(center, radius, n_samples)
    model.eval()
    with torch.no_grad():
        x_center = model.decode_deterministic(center.view(1, -1))
        x_samples = model.decode_deterministic(z_samples)
        x_center_score = score_function(x_center, target)
        x_samples_scores = score_function(x_samples, target)
    return np.abs(x_center_score - x_samples_scores).mean()


def make_local_stationarity_plots(
        centers: Tensor,
        radiuses: list,
        n_samples: int,
        model: Union[EquationGrammarModelTorch, ShapesVAE, TopologyVAE],
        score_function: Callable,
        target: Union[Tensor, np.ndarray],
        save_dir: str,
        dist: Optional[str] = "l2",
):
    # sample all points in once and then dispatch them
    z_dim = centers.shape[-1]
    n_centers = centers.shape[0]
    n_radiuses = len(radiuses)
    n_total_samples = int(n_centers * n_radiuses * n_samples)
    if dist == "sup":
        all_samples = sample_on_hypercube(
            center=torch.zeros_like(centers[0]),
            side=1,
            n_samples=n_total_samples
        )
    elif dist == "l1":
        all_samples = sample_on_hyperdiamond(
            center=torch.zeros_like(centers[0]),
            side=1,
            n_samples=n_total_samples
        )
    else:
        all_samples = sample_on_hypersphere(
            center=torch.zeros_like(centers[0]),
            radius=1,
            n_samples=n_total_samples
        )
    all_samples_resize = all_samples.view(n_centers, n_radiuses, n_samples, -1)
    all_samples_radius = torch.tensor(radiuses).view(1, -1, 1, 1).to(centers) * all_samples_resize
    all_samples_center = all_samples_radius + centers.view(n_centers, 1, 1, -1)
    all_samples = all_samples_center.view(-1, z_dim)

    model.eval()
    model.to(centers.device) if not isinstance(model, EquationGrammarModelTorch) else model.vae.to(centers.device)
    res = np.zeros((n_centers, n_radiuses))
    with torch.no_grad():
        if isinstance(model, EquationGrammarModelTorch):
            dec_cntrs = model.decode_from_latent_space(zs=centers, n_decode_attempts=100)
            dec_cntrs_scores = score_function(dec_cntrs)
            bs = n_radiuses * n_samples
            dataset = TensorDataset(all_samples)
            dl = DataLoader(dataset, batch_size=bs)
            i = 0
            for (batch,) in tqdm(dl):
                dec_batch = model.decode_from_latent_space(batch)
                dec_batch_sores = score_function(dec_batch)
                dec_batch_scores_gap = np.abs(dec_cntrs_scores[i] - dec_batch_sores)
                dec_batch_scores_gap_mean_rad = dec_batch_scores_gap.reshape(n_radiuses, n_samples).mean(-1)
                res[i] = dec_batch_scores_gap_mean_rad
                i += 1
        else:
            dec_cntrs = model.decode_deterministic(centers)
            dec_cntrs_scores = score_function(dec_cntrs, target)
            bs = n_radiuses * n_samples
            dataset = TensorDataset(all_samples)
            dl = DataLoader(dataset, batch_size=bs)
            i = 0
            for (batch,) in tqdm(dl):
                dec_batch = model.decode_deterministic(batch)
                dec_batch_sores = score_function(dec_batch, target)
                dec_batch_scores_gap = np.abs(dec_cntrs_scores[i] - dec_batch_sores)
                dec_batch_scores_gap_mean_rad = dec_batch_scores_gap.reshape(n_radiuses, n_samples).mean(-1)
                res[i] = dec_batch_scores_gap_mean_rad
                i += 1

    # loop version is slow but doesn't run out of memory
    # res = np.zeros((n_centers, n_radiuses))
    # for i, c in enumerate(centers):
    #     for j, r in enumerate(radiuses):
    #         res[i][j] = average_distance_to_center_target_value(
    #             center=c,
    #             radius=r,
    #             model=model,
    #             score_function=score_function,
    #             target=target,
    #             samples=all_samples[i][j]
    #         )

    plt.imshow(res)
    plt.title(f"Avg. gap to target score per $z$ & per {dist.capitalize()}-dist radius")
    plt.xlabel(f"radius of {dist.capitalize()}-ball around center z")
    plt.ylabel("centers ($z \in$ train set)")
    tickidx = np.arange(0, len(radiuses), len(radiuses) // 10)
    plt.xticks(tickidx, np.round(radiuses[tickidx]))
    plt.savefig(os.path.join(save_dir, f"local_stationarity_{dist}.pdf"))
    plt.close()

    plt.plot(res.mean(0))
    plt.fill_between(np.arange(n_radiuses), res.mean(0) + res.std(0), res.mean(0) - res.std(0), alpha=0.2)
    plt.title(f"Avg. gap to target score (avg. on {res.shape[0]} latent points) per {dist}-dist radius")
    plt.xlabel(f"radius of {dist.capitalize()}-ball around center z")
    tickidx = np.arange(0, len(radiuses), len(radiuses) // 10)
    plt.xticks(tickidx, np.round(radiuses[tickidx]))
    plt.savefig(os.path.join(save_dir, f"local_stationarity_average_{dist}.pdf"))
    plt.close()

    # get all pairwise distances in latent space and in score space then sort and plot one versus the other
    lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
    y_distance_matrix = lpydist(torch.from_numpy(dec_cntrs_scores).to(centers))
    y_dists_tril_idx = torch.tril_indices(y_distance_matrix.shape[0], y_distance_matrix.shape[1], offset=-1)
    y_dists_tril = y_distance_matrix[y_dists_tril_idx[0, :], y_dists_tril_idx[1, :]]
    y_dists_sorted, sorted_idx = torch.sort(y_dists_tril)

    for dist in ['sup', 'l2', 'l1', 'cos']:
        if dist == "l2":
            lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        elif dist == "l1":
            lpembdist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        elif dist == "cos":
            lpembdist = distances.DotProductSimilarity()
        else:
            lpembdist = distances.LpDistance(normalize_embeddings=False, p=np.inf, power=1)

        emb_distance_matrix = lpembdist(centers)
        emb_dists_tril_idx = torch.tril_indices(emb_distance_matrix.shape[0], emb_distance_matrix.shape[1], offset=-1)
        emb_dists_tril = emb_distance_matrix[emb_dists_tril_idx[0, :], emb_dists_tril_idx[1, :]]
        emb_dists_sorted = emb_dists_tril[sorted_idx]

        dy, dz = y_dists_sorted.cpu().numpy(), emb_dists_sorted.cpu().numpy()
        plt.scatter(dy, dz, marker="+", alpha=0.25)
        # plt.fill_between(dz.mean(), )
        plt.title(f"")
        plt.xlabel(f"absolute difference in score")
        plt.ylabel(f"{dist}-distance in latent space")
        plt.savefig(os.path.join(save_dir, f"y-dist_vs_z_{dist}-dist.pdf"))
        plt.close()

        y_dists_sorted_cat = y_dists_sorted.view(-1, len(y_dists_sorted) // 100).cpu().numpy()
        emb_dists_sorted_cat = emb_dists_sorted.view(-1, len(emb_dists_sorted) // 100).cpu().numpy()
        plt.plot(y_dists_sorted_cat.mean(-1), emb_dists_sorted_cat.mean(-1))
        plt.fill_between(y_dists_sorted_cat.mean(-1),
                         emb_dists_sorted_cat.mean(-1) + emb_dists_sorted_cat.std(-1),
                         emb_dists_sorted_cat.mean(-1) - emb_dists_sorted_cat.std(-1),
                         alpha=0.2)
        plt.title(f"Avg. {dist}-dist in latent space vs. avg. absolute score gap")
        plt.xlabel(f"avg. absolute difference in score")
        plt.ylabel(f"avg. {dist}-distance in latent space")
        plt.savefig(os.path.join(save_dir, f"y-dist_vs_z_{dist}-dist_cat.pdf"))
        plt.close()

    print("Local stationarity plots done.")
