import model_utils.input_transform as tfx

from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Callable


def safe_scaling(y, scaler=None, scaling_method="standardize"):
    """
    Apply scaling to the target
    """
    try:
        if scaler is None:  # fit a new scaler
            if scaling_method == "standardize":
                scaler = StandardScaler()
                y_scaled = scaler.fit_transform(y)
            elif scaling_method == "power_transform":
                if y.min() <= 0:
                    scaler = PowerTransformer(method='johnson', standardize=True)
                    y_scaled = scaler.fit_transform(y)
                else:
                    scaler = PowerTransformer(method='box-cox', standardize=True)
                    y_scaled = scaler.fit_transform(y)
                    # try johnson
                    if y_scaled.std() < 0.5:
                        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                        y_scaled = scaler.fit_transform(y)
                    if y_scaled.std() < 0.5:
                        raise RuntimeError('Power transformation failed')
            elif scaling_method == "min_max":
                scaler = MinMaxScaler()
                y_scaled = scaler.fit_transform(y)
            else:
                raise ValueError("Unknown scaling method:", scaling_method)
        else:  # transform using the given scaler
            y_scaled = scaler.transform(y)
    except Exception as e:
        print(f"[Warn] scaling fails:", e)
        y_scaled = y.copy()
        scaler = None
    return y_scaled, scaler


def filter_nan(x, xe, y, keep_rule='any'):
    assert x is None or np.isfinite(x).all()
    assert xe is None or np.isfinite(xe).all()
    assert torch.isfinite(y).any(), "No valid data in the dataset"

    if keep_rule == 'any':
        valid_id = torch.isfinite(y).any(dim=1)
    else:
        valid_id = torch.isfinite(y).all(dim=1)
    x_filtered = x[valid_id] if x is not None else None
    xe_filtered = xe[valid_id] if xe is not None else None
    y_filtered = y[valid_id]
    return x_filtered, xe_filtered, y_filtered


def get_gp_prediction(model, x, scaler, **kwargs):
    pred = model.predict(x, **kwargs)
    pred_lcb, pred_ucb = pred.confidence_region()
    pred_mean = pred.mean
    if scaler is not None:
        mean = scaler.inverse_transform(pred_mean.detach().numpy().reshape(-1, 1)).flatten()
        lcb = scaler.inverse_transform(pred_lcb.detach().numpy().reshape(-1, 1)).flatten()
        ucb = scaler.inverse_transform(pred_ucb.detach().numpy().reshape(-1, 1)).flatten()
    else:
        mean = pred_mean.detach().numpy().flatten()
        lcb = pred_lcb.detach().numpy().flatten()
        ucb = pred_ucb.detach().numpy().flatten()
    return pred, mean, lcb, ucb


class OneHotTransform(torch.nn.Module):
    def __init__(self, num_uniqs):
        super().__init__()
        self.num_uniqs = num_uniqs

    @property
    def num_out(self) -> int:
        return sum(self.num_uniqs)

    def forward(self, xe):
        return torch.cat(
            [torch.nn.functional.one_hot(xe[:, i].long(), self.num_uniqs[i])
             for i in range(xe.shape[1])], dim=1
        ).float()


class EmbTransform(nn.Module):
    def __init__(self, num_uniqs, **conf):
        super().__init__()
        self.emb_sizes = conf.get('emb_sizes')
        if self.emb_sizes is None:
            self.emb_sizes = [min(50, 1 + v // 2) for v in num_uniqs]

        self.emb = nn.ModuleList([])
        for num_uniq, emb_size in zip(num_uniqs, self.emb_sizes):
            self.emb.append(nn.Embedding(num_uniq, emb_size))

    @property
    def num_out(self) -> int:
        return sum(self.emb_sizes)

    def forward(self, xe):
        return torch.cat(
            [self.emb[i](xe[:, i]).view(xe.shape[0], -1) for i in range(len(self.emb))], dim=1)


def get_model_prediction(model, Xc_te, support_decomposed_pred):
    """
    Given a model and test data, return model predictions
    :param model:
    :param Xc_te:
    :param support_decomposed_pred: whether to return a list of decomposed prediction
    :return:
    """
    preds = []
    # full prediction
    if support_decomposed_pred:
        py_m0, ps2_m0 = model.predict(
            torch.FloatTensor(Xc_te), torch.zeros(Xc_te.shape[0], 0), with_noise=True, mode=0
        )
    else:
        py_m0, ps2_m0 = model.predict(torch.FloatTensor(Xc_te), torch.zeros(Xc_te.shape[0], 0))
    ucb_m0 = py_m0 + (torch.sqrt(ps2_m0) * 2.0)
    lcb_m0 = py_m0 - (torch.sqrt(ps2_m0) * 2.0)
    preds.append(
        (py_m0.detach().numpy(),
         ps2_m0.detach().numpy(),
         lcb_m0.detach().numpy(),
         ucb_m0.detach().numpy(),
         )
    )

    if support_decomposed_pred:
        # mode 1 predict
        py_m1, ps2_m1 = model.predict(
            torch.FloatTensor(Xc_te), torch.zeros(Xc_te.shape[0], 0), with_noise=True, mode=1
        )
        ucb_m1 = py_m1 + (torch.sqrt(ps2_m1) * 2.0)
        lcb_m1 = py_m1 - (torch.sqrt(ps2_m1) * 2.0)
        preds.append(
            (py_m1.detach().numpy(),
             ps2_m1.detach().numpy(),
             lcb_m1.detach().numpy(),
             ucb_m1.detach().numpy(),
             )
        )

        # mode 2 predict
        py_m2, ps2_m2 = model.predict(
            torch.FloatTensor(Xc_te), torch.zeros(Xc_te.shape[0], 0), with_noise=True, mode=2
        )
        ucb_m2 = py_m2 + (torch.sqrt(ps2_m2) * 2.0)
        lcb_m2 = py_m2 - (torch.sqrt(ps2_m2) * 2.0)
        preds.append(
            (py_m2.detach().numpy(),
             ps2_m2.detach().numpy(),
             lcb_m2.detach().numpy(),
             ucb_m2.detach().numpy(),
             )
        )
    return preds


def get_kernel_lengthscale(kern_model):
    # get lengthscale
    ls = kern_model.lengthscale
    km = kern_model
    while ls is None and getattr(km, 'base_kernel', None) is not None:
        km = km.base_kernel
        ls = km.lengthscale
    if ls is not None:
        ls = ls.detach().cpu().numpy().flatten()

    if isinstance(ls, Iterable) and len(ls) >= 1:
        str_output = [f'{i:.3f}' for i in ls] if len(ls) > 1 else f'{ls[0]:.3f}'
    else:
        str_output = f'{ls}'

    return ls, str_output


def get_kernel_output_scale(kernel):
    oscale = kernel.outputscale.item() if hasattr(kernel, 'outputscale') else None
    oscale_str = f'{oscale:.3f}' if oscale is not None else f'{oscale}'
    return oscale, oscale_str



def prepare_data(input_type: str,
                 n_var: int, raw_input_mean: [float, np.array], raw_input_std: [float, np.array],
                 xc_sample_size: int, input_sampling_func: Callable,
                 xc_raw: np.array, y: np.array,
                 dtype: torch.dtype, device: torch.device,
                 **data_cfg):
    """
    Prepare the data acd. to the input type, transform them into tensor and put on right device
    """
    if input_type == INPUT_TYPE_NOISED or input_type == INPUT_TYPE_MEAN:  # only the x_raw
        x_ts = torch.tensor(xc_raw, dtype=dtype, device=device)
    elif input_type == INPUT_TYPE_SAMPLES:  # observe some samples around x
        tf_add_xsamp = tfx.AdditionalFeatures(
            f=tfx.additional_xc_samples, transform_on_train=False,
            fkwargs={'n_sample': xc_sample_size, 'n_var': n_var,
                     'sampling_func': input_sampling_func}
        )
        x_ts = tf_add_xsamp.transform(torch.tensor(xc_raw, dtype=dtype, device=device))
    elif input_type == INPUT_TYPE_DISTRIBUTION:
        tf_add_std = tfx.AdditionalFeatures(f=tfx.additional_std, transform_on_train=False,
                                            fkwargs={'std': raw_input_std})
        x_ts = tf_add_std.transform(
            torch.tensor(
                xc_raw + raw_input_mean if raw_input_mean is not None else xc_raw,
                dtype=dtype, device=device
            ) # we add mean here
        )

    else:
        raise ValueError('Unknown input type:', input_type)

    y_ts = torch.tensor(y.reshape(-1, 1), dtype=dtype, device=device)

    return x_ts, y_ts


NOISE_LB = 1e-4
INPUT_TYPE_NOISED = 'exact_input'
INPUT_TYPE_MEAN = 'mean_input'
INPUT_TYPE_SAMPLES = 'sample_input'
INPUT_TYPE_DISTRIBUTION = 'distribution_input'
