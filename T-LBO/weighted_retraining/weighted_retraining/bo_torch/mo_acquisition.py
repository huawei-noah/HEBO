# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Dict, Iterable

import numpy as np
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.utils import _filter_kwargs
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

from weighted_retraining.weighted_retraining.bo_torch import mo_acq_func
from weighted_retraining.weighted_retraining.bo_torch.optimize import optimize_acqf_torch


def query_acq_func(acq_func_id: str, acq_func_kwargs: Dict[str, Any],
                   gp_model: SingleTaskGP, gp_model_error: SingleTaskGP, vae_model,
                   q: int, num_MC_samples_acq: int):
    if not hasattr(AnalyticAcquisitionFunction, acq_func_id):
        # use MC version of acq function
        acq_func_id = f'q{acq_func_id}'
        resampler = SobolQMCNormalSampler(num_samples=num_MC_samples_acq, resample=True).to(gp_model.train_inputs[0])
        acq_func_kwargs['sampler'] = resampler
        acq_func_class = getattr(mo_acq_func, acq_func_id)

        error_resampler = SobolQMCNormalSampler(num_samples=num_MC_samples_acq, resample=True).to(
            gp_model_error.train_inputs[0])
        acq_func_kwargs['error_sampler'] = error_resampler
        acq_func = acq_func_class(gp_model, gp_model_error, **_filter_kwargs(acq_func_class, **acq_func_kwargs))

    return acq_func


def bo_mo_loop(gp_model: SingleTaskGP, gp_model_error: SingleTaskGP, vae_model,
               acq_func_id: str, acq_func_kwargs: Dict[str, Any], acq_func_opt_kwargs: Dict[str, Any],
               bounds: Tensor, tkwargs: Dict[str, Any], q: int,
               num_restarts: int, raw_initial_samples, seed: int, num_MC_sample_acq: int) -> Iterable[Tensor]:
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)

    # put on proper device

    # we want to maximize
    fmax = torch.quantile(gp_model.train_targets, .9).item()
    print(f"Using good point cutoff {fmax:.2f}")
    ref_point = np.array([
        torch.quantile(gp_model.train_targets, .1).item(),
        torch.quantile(gp_model_error.train_targets, .1).item()])
    print(f"Using ref point cutoff {ref_point}")
    acq_func_kwargs['ref_point'] = ref_point

    device = gp_model.train_inputs[0].device

    bounds = bounds.to(**tkwargs)
    gp_model.eval()
    gp_model_error.eval()

    acq_func_kwargs['best_f'] = fmax
    acq_func = query_acq_func(acq_func_id=acq_func_id, acq_func_kwargs=acq_func_kwargs,
                              gp_model=gp_model, gp_model_error=gp_model_error, vae_model=vae_model,
                              q=q, num_MC_samples_acq=num_MC_sample_acq
                              )
    acq_func.to(**tkwargs)

    options = {'batch_limit': 100, "maxiter": 500} if acq_func_opt_kwargs == {} else acq_func_opt_kwargs
    print("Start acquisition function optimization...")
    if q == 1 and isinstance(acq_func, AnalyticAcquisitionFunction):
        # use optimize_acq (with LBFGS)
        candidate, acq_value = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=num_restarts,
                                             raw_samples=raw_initial_samples, return_best_only=True, options=options
                                             )
    else:
        candidate, acq_value = optimize_acqf_torch(
            acq_function=acq_func, bounds=bounds, q=q, num_restarts=num_restarts, raw_samples=raw_initial_samples,
            return_best_only=True, verbose=True, options=options
        )
    print(f"Acquired {candidate} with acquisition value {acq_value}")

    return candidate.to(device=device)
