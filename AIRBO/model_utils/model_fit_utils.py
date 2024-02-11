import gpytorch as gpyt
import numpy as np
import torch
from typing import Dict, List, Tuple


def fit_model_restarts(model_cls: torch.nn.Module, model_cfg: Dict,
                       tr_Xc: torch.Tensor, tr_Xe: torch.Tensor, tr_y:torch.Tensor,
                       fit_cfg: Dict) -> Tuple[torch.nn.Module, List]:
    """
    Train a GP model with restarts (if NotPSD Error happens)
    :param model_cls: model class
    :param model_cfg: model configuration
    :param tr_Xc: continuous inputs
    :param tr_Xe: enumerate inputs
    :param tr_y: training target
    :param fit_cfg: fitting configurations
    :return: a list of training history
    """
    torch.autograd.set_detect_anomaly(True)
    fit_restarts = fit_cfg.get('fit_restarts', 3)
    results = []
    while len(results) < fit_restarts:
        try:
            model_i = model_cls(**model_cfg)
            tr_hist_i = model_i.fit(
                tr_Xc,
                tr_Xe,
                tr_y.flatten(),
                **fit_cfg
            )
        except gpyt.utils.errors.NotPSDError as e:
            print('[WARN] model fit fails, try again:', e)
            continue
        results.append((model_i, tr_hist_i))

    if len(results) > 0:
        # find the optimal model
        opt_ind = np.nanargmin([h[-1]['loss'] for (m, h) in results])
        opt_model, opt_tr_hist = results[opt_ind]
    else:
        raise ValueError(f"All the {fit_restarts} restarts fail!")

    return opt_model, opt_tr_hist