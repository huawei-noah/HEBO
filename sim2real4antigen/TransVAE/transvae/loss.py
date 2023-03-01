from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch import Tensor
from torch.autograd import Variable
from pytorch_metric_learning import distances

class ContrastiveLossTorch:

    def __init__(self, threshold: float, hard: Optional[bool] = None):
        self.threshold = threshold
        self.hard = hard if hard is not None else False

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)
        # emb_distance_matrix = torch.cdist(embs[:, None], embs[None, :], p=2)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)
        # y_distance_matrix = torch.cdist(ys[:, None], ys[None, :], p=1)

        loss = torch.zeros_like(emb_distance_matrix).to(embs)

        threshold_matrix = self.threshold * torch.ones(loss.shape).to(embs)

        high_dy_filter = y_distance_matrix > self.threshold
        aux_max_dz_thr = torch.maximum(emb_distance_matrix, threshold_matrix)
        aux_min_dz_thr = torch.minimum(emb_distance_matrix, threshold_matrix)

        if self.hard:
            # dy - dz
            loss[high_dy_filter] = y_distance_matrix[high_dy_filter] - emb_distance_matrix[high_dy_filter]
            # dz
            loss[~high_dy_filter] = emb_distance_matrix[~high_dy_filter]
        else:
            # (2 - min(threshold, dz) / threshold) * (dy - max(dz, threshold))
            loss[high_dy_filter] = (2 - aux_min_dz_thr[high_dy_filter]).div(self.threshold) * (
                    y_distance_matrix[high_dy_filter] - aux_max_dz_thr[high_dy_filter])

            #  max(threshold, dz) / threshold * (min(dz, threshold) - dy)
            loss[~high_dy_filter] = aux_max_dz_thr[~high_dy_filter].div(self.threshold) * (
                    aux_min_dz_thr[~high_dy_filter] - y_distance_matrix[~high_dy_filter])

        loss = torch.relu(loss)
        return loss

    def compute_loss(self, embs: Tensor, ys: Tensor):
        loss_matrix = torch.triu(self.build_loss_matrix(embs, ys), diagonal=1)
        n = (loss_matrix > 0).sum()

        if n == 0:
            n = 1
        # average over non-zero elements
        return loss_matrix.sum().div(n)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, hard: Optional[bool] = None) -> str:
        metric_id = f'contrast-thr-{threshold:g}'
        if hard:
            metric_id += '-hard'
        return metric_id


class TripletLossTorch:
    def __init__(self, threshold: float, margin: Optional[float] = None, soft: Optional[bool] = False,
                 eta: Optional[float] = None):
        """
        Compute Triplet loss
        Args:
            threshold: separate positive and negative elements in temrs of `y` distance
            margin: hard triplet loss parameter
            soft: whether to use sigmoid version of triplet loss
            eta: parameter of hyperbolic function softening transition between positive and negative classes
        """
        self.threshold = threshold
        self.margin = margin
        self.soft = soft
        assert eta is None or eta > 0, eta
        self.eta = eta

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        # zd = emb_distance_matrix.detach().cpu().numpy().flatten()
        # yd = y_distance_matrix.detach().cpu().numpy().flatten()
        # plt.hist(zd, bins=50)
        # plt.savefig(os.path.join(os.getcwd(), 'z_dist.pdf'))
        # plt.close()
        # plt.hist(yd, bins=50)
        # plt.savefig(os.path.join(os.getcwd(), 'y_dist.pdf'))
        # plt.close()

        positive_embs = emb_distance_matrix.where(y_distance_matrix <= self.threshold, torch.tensor(0.).to(embs))
        negative_embs = emb_distance_matrix.where(y_distance_matrix > self.threshold, torch.tensor(0.).to(embs))

        loss_loop = 0 * torch.tensor([0.], requires_grad=True).to(embs)
        n_positive_triplets = 0
        pos, neg = [], []
        for i in range(embs.size(0)):
            pos_i = positive_embs[i][positive_embs[i] > 0]
            neg_i = negative_embs[i][negative_embs[i] > 0]
            pos.append(pos_i.detach().cpu().numpy())
            neg.append(neg_i.detach().cpu().numpy())
            pairs = torch.cartesian_prod(pos_i, -neg_i)
            if self.soft:
                triplet_losses_for_anchor_i = torch.nn.functional.softplus(pairs.sum(dim=-1))
                if self.eta is not None:
                    # get the corresponding delta ys
                    pos_y_i = y_distance_matrix[i][positive_embs[i] > 0]
                    neg_y_i = y_distance_matrix[i][negative_embs[i] > 0]
                    pairs_y = torch.cartesian_prod(pos_y_i, neg_y_i)
                    assert pairs.shape == pairs_y.shape, (pairs_y.shape, pairs.shape)
                    triplet_losses_for_anchor_i = triplet_losses_for_anchor_i * \
                                                  self.smooth_indicator(self.threshold - pairs_y[:, 0]) \
                                                      .div(self.smooth_indicator(self.threshold)) \
                                                  * self.smooth_indicator(pairs_y[:, 1] - self.threshold) \
                                                      .div(self.smooth_indicator(1 - self.threshold))
            else:
                triplet_losses_for_anchor_i = torch.relu(self.margin + pairs.sum(dim=-1))
            n_positive_triplets += (triplet_losses_for_anchor_i > 0).sum()
            loss_loop += triplet_losses_for_anchor_i.sum()
        loss_loop = loss_loop.div(max(1, n_positive_triplets))

        pos = np.concatenate(pos)
        neg = np.concatenate(neg)
        # plt.hist(pos, bins=50, alpha=0.2, label='pos')
        # plt.hist(neg, bins=50, alpha=0.2, label='neg')
        # plt.legend()
        # plt.title("Distances between positive and negative embeddings")
        # plt.savefig(os.path.join(os.getcwd(), 'neg_pos_dist.pdf'))
        # plt.close()

        return loss_loop

    def smooth_indicator(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, float):
            return np.tanh(x / (2 * self.eta))
        return torch.tanh(x / (2 * self.eta))

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, margin: Optional[float] = None, soft: Optional[bool] = None,
                      eta: Optional[bool] = None) -> str:
        if margin is not None:
            return f'triplet-thr-{threshold:g}-mrg-{margin:g}'
        if soft is not None:
            metric_id = f'triplet-thr-{threshold:g}-soft'
            if eta is not None:
                metric_id += f'-eta-{eta:g}'
            return metric_id


class LogRatioLossTorch:
    def __init__(self):
        """
        Compute Log-ration loss (https://arxiv.org/pdf/1904.09626.pdf)
        """
        pass

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        eps = 1e-4 / embs.size(0)

        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=2)
        emb_distance_matrix = torch.sqrt(lpembdist(embs) + eps)  # L2dist

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        eps = 1e-6

        loss_loop = 0 * torch.tensor([0.], requires_grad=True).to(embs)
        n_positive_triplets = 0
        m = embs.size()[0] - 1  # #paired

        # auxiliary variable for uniform weighting of triplets (a, i, j) where i < j
        wgt = torch.arange(1, m + 1).to(embs)
        wgt = wgt.repeat(m, 1).t() < wgt.repeat(m, 1)
        wgt = wgt.div(wgt.sum())

        for ind_a in range(embs.size(0)):
            # auxiliary variables
            idxs = torch.arange(0, m).to(device=embs.device)
            idxs[ind_a:] += 1

            log_dist = torch.log(emb_distance_matrix[ind_a][idxs] + eps)
            log_y_dist = torch.log(y_distance_matrix[ind_a][idxs] + eps)

            diff_log_dist = log_dist.repeat(m, 1).t() - log_dist.repeat(m, 1)
            diff_log_y_dist = log_y_dist.repeat(m, 1).t() - log_y_dist.repeat(m, 1)
            assert diff_log_y_dist.shape == diff_log_dist.shape == (m, m), (diff_log_y_dist.shape,
                                                                            diff_log_dist.shape, m)
            valid_aij = diff_log_y_dist < 0  # keep triplet having D(y_a, y_i) < D(y_q, y_j)

            log_ratio_loss = (diff_log_dist - diff_log_y_dist).pow(2)[valid_aij].sum()

            loss_loop += log_ratio_loss
            n_positive_triplets += valid_aij.sum()

        loss_loop = loss_loop.div(max(1, n_positive_triplets))

        return loss_loop

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id() -> str:
        metric_id = "log-ratio"
        return metric_id


def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + MSE, BCE, KLD, MSE

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + MSE, BCEmol, BCEmask, KLD, MSE


def trans_vae_fixed_len_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1,
                             metric_learning: bool = False, metric='contrastive', threshold=0.1, embs=None):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))

    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)

    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)

    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)

    if metric_learning:
        if metric == 'contrastive':
            MetricLossClass = ContrastiveLossTorch(threshold=threshold)
        elif metric == 'triplet':
            MetricLossClass = ContrastiveLossTorch(threshold=threshold)
        elif metric == 'logratio':
            MetricLossClass = ContrastiveLossTorch(threshold=threshold)
        else:
            raise ValueError(f'unknown metric {metric}')
        MetricLoss = MetricLossClass(embs=embs, ys=true_prop.reshape(-1, 1))
    else:
        MetricLoss = torch.tensor(0.)

    return BCEmol + KLD + MSE + MetricLoss, BCEmol, None, KLD, MSE, MetricLoss
