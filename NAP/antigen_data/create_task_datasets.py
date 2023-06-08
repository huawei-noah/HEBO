# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import glob

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskVariationalGP, SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import PredictiveLogLikelihood
from botorch.optim.fit import fit_gpytorch_mll_torch
import botorch


from pathlib import Path
import os, sys
ROOT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT)

from nap.RL.utils_gp import TransformedCategorical


def clean(seq_string):
    return list(map(int, seq_string.split(',')))


if __name__ == '__main__':
    antigen_data_root = os.path.join(ROOT, 'antigen_data')

    antigen_datasets_paths = os.listdir(antigen_data_root)
    antigen_datasets_paths = [a for a in antigen_datasets_paths if '.csv' in a]
    np.random.shuffle(antigen_datasets_paths)
    for i, antigen in enumerate(antigen_datasets_paths):
        print(i, 'Antigen', antigen)
        # Fit and save GP
        antigen_dataset_path = os.path.join(antigen_data_root, antigen)
        antigen_gp_path = os.path.join(antigen_data_root, f'{antigen.split(".")[0]}.pt')
        if os.path.exists(antigen_gp_path):
            print('Already computed GP')
            continue

        antigen_dataset = pd.read_csv(antigen_dataset_path, converters={'domain': clean})
        tokenized_seq = antigen_dataset['domain'].values

        X = torch.from_numpy(np.stack(tokenized_seq))
        stdY = torch.from_numpy(antigen_dataset['accs'].values)
        print(f'{i} Antigen {antigen} has {len(stdY)} points.')
        X = X.to(dtype=float, device='cuda')
        stdY = stdY.to(dtype=float, device='cuda')

        model = SingleTaskVariationalGP(
            train_X=X,
            train_Y=stdY.view(-1, 1),
            covar_module=TransformedCategorical(ard_num_dims=X.shape[-1]).cuda(),
            inducing_points=int(0.1 * X.shape[0])
        ).cuda()
        mll = PredictiveLogLikelihood(likelihood=model.likelihood, model=model.model, num_data=len(stdY))

        # model = SingleTaskGP(
        #     train_X=X,
        #     train_Y=stdY.view(-1, 1),
        #     covar_module=TransformedCategorical(ard_num_dims=X.shape[-1]),
        # )
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            mll.cuda()
            _ = fit_gpytorch_mll_torch(mll)
        except (RuntimeError, botorch.exceptions.errors.ModelFittingError) as e:
            print(e)
            try:
                print('Try fit on CPU')
                mll.cpu()
                _ = fit_gpytorch_mll(mll=mll)
            except RuntimeError as e:
                print(f'Error during the GP fit on {antigen}.')
                print(e)
                del antigen_dataset, tokenized_seq, X, stdY, model, mll
                torch.cuda.empty_cache()
                continue

        with torch.no_grad():
            torch.save(model, antigen_gp_path)
            print(f'saved {antigen_gp_path}')

        del antigen_dataset, tokenized_seq, X, stdY, model, mll
        torch.cuda.empty_cache()
