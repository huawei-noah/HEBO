# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import json
import pickle

import numpy as np

# based on https://arxiv.org/pdf/2106.06257.pdf

#wget https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip

ids = {
    "5860": "glmnet",
    "4796": "rpart_preproc",
    "5906": "xgboost",
    "5859": "rpart",
    "5889": "ranger",
    "5527": "svm",
}

for dstype in ['test', 'train', 'validation']:
    dataset_name = f"meta-train-dataset-augmented.json" if 'train' in dstype else f"meta-{dstype}-dataset.json"
    with open(dataset_name, "r") as f:
        data = json.load(f)
        for space_id, label in ids.items():
            index = 0
            for dataset_key in data[space_id].keys():

                hpo_format = dict()
                hpo_format["domain"] = np.array(data[space_id][dataset_key]["X"])
                hpo_format["accs"] = np.array(data[space_id][dataset_key]["y"])[..., 0]

                assert hpo_format["accs"].max() <= 1.0
                assert hpo_format["accs"].min() >= 0.0

                path = f"{label}_{dstype}_{index}.pkl"
                print(path)

                if index == 0 and dstype == "test":
                    print(f"problem {label} dim {hpo_format['domain'].shape[1]}")
                elif dstype == "train":
                    print("number of points", hpo_format["accs"].shape[0], "min y",
                          hpo_format["accs"].min(), "max y", hpo_format["accs"].max())

                with open(path, 'wb') as f:
                    pickle.dump(hpo_format, f)

                index += 1
