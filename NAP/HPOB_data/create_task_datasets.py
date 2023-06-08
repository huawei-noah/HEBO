# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import json
import os
import pickle as pkl

import botorch
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood

from pathlib import Path
import os, sys
ROOT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT)

if __name__ == '__main__':

    models = ['glmnet', 'rpart_preproc', 'xgboost', 'ranger', 'rpart', 'svm']
    hpob_data_root = os.path.join(ROOT, 'HPOB_data')

    if not os.path.exists(os.path.join(hpob_data_root, 'gps')):
        os.makedirs(os.path.join(hpob_data_root, 'gps'))

    name_ids = {
        "glmnet": "5860",
        "rpart_preproc": "4796",
        "xgboost": "5906",
        "ranger": "5889",
        "rpart": "5859",
        "svm": "5527",
    }

    with open(os.path.join(hpob_data_root, "meta-dataset-descriptors.json")) as f:
        descriptor = json.load(f)

    for model_name in models:
        search_space_id = name_ids[model_name]
        search_space_desc = descriptor[search_space_id]
        train_datasets = os.listdir(os.path.join(hpob_data_root))
        train_datasets = sorted([d for d in train_datasets if model_name + '_train' in d and 'pkl' in d])

        skipped, skipped_n = [], []
        for dataset in train_datasets:
            data = pkl.load(open(os.path.join(hpob_data_root, dataset), 'rb'))
            Y = data['accs']
            stdY = (Y - Y.mean()) / Y.std()

            if np.isnan(stdY).any():
                print(f"({model_name}) Dataset #{dataset} Y.std()=NaN Skipped")
                skipped.append(dataset)
                skipped_n.append(int(dataset.split(".pkl")[0].split("_")[-1]))
                continue
            if Y.std() < 1e-3:
                print(f"({model_name}) Dataset #{dataset} Y.std()={Y.std():.10f} Skipped")
                skipped.append(dataset)
                skipped_n.append(int(dataset.split(".pkl")[0].split("_")[-1]))
                continue

        print(f"({model_name}) skipped datasets {skipped_n}")
        train_datasets = [trd for trd in train_datasets if trd not in skipped]

        for dataset in train_datasets:
            gp_name = dataset.split('.pkl')[0] + f'_gp.pt'
            if not os.path.exists(os.path.join(hpob_data_root, 'gps', gp_name)):
                data = pkl.load(open(os.path.join(hpob_data_root, dataset), 'rb'))
                Y = data['accs']
                X = data['domain']  # X is already normalised across all datasets (train, val, test)

                yuniq, ycount = np.unique(Y, return_counts=True)
                counts = {v: c for v, c in zip(yuniq, ycount)}
                logits = np.array([Y[i] / counts[Y[i]] for i in range(len(Y))])
                freq_idx = logits.argsort()[::-1]

                selected_rows = freq_idx[:(3 * len(yuniq))]
                np.random.shuffle(selected_rows)
                X = X[selected_rows]
                Y = Y[selected_rows]
                stdY = (Y - Y.mean()) / Y.std()

                num_dims = list(np.arange(X.shape[-1]))
                cat_dims = []

                # Fit and save GP
                print(f'Fit GP on dataset {dataset} containing {X.shape[0]} points...')
                normX = torch.from_numpy(X).to(dtype=torch.float64)
                stdY = torch.from_numpy(stdY).to(dtype=torch.float64)

                # Sub-sample dataset
                model = SingleTaskGP(train_X=normX, train_Y=stdY.view(-1, 1))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                try:
                    mll.cpu()
                    _ = fit_gpytorch_mll(mll=mll)
                except (RuntimeError, botorch.exceptions.errors.ModelFittingError) as e:
                    print(e)
                    try:
                        print('Try fit on GPU')
                        mll.cuda()
                        _ = fit_gpytorch_mll_torch(mll)
                    except RuntimeError as e:
                        print(f'Error during the GP fit on {dataset}.')
                        print(e)
                        normX = normX.cpu().numpy()
                        stdY = stdY.cpu().numpy()
                        model = model.cpu()
                        mll = mll.cpu()
                        del model, mll
                        torch.cuda.empty_cache()
                        continue

                with torch.no_grad():
                    torch.save(model, os.path.join(hpob_data_root, 'gps', gp_name))
                print(f"saved model at {os.path.join(hpob_data_root, 'gps', gp_name)}")

                normX = normX.cpu()
                stdY = stdY.cpu()
                model = model.cpu()
                mll = mll.cpu()
                model.eval()
                del normX, stdY, model, mll
                torch.cuda.empty_cache()

            else:
                data = pkl.load(open(os.path.join(hpob_data_root, dataset), 'rb'))
                X = data['domain']
                print(f'{dataset} GP already fit and saved: {X.shape[0]} points in {X.shape[1]} dims.')
