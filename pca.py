# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from dataclasses import dataclass, field
import torch
from torch_pca import PCA
from transformers import set_seed, HfArgumentParser


@dataclass
class PCAArguments:
    dataset: str = field(default="EvolInstructCode", metadata={"help": "Dataset name for the PCA model."})
    model_name: str = field(default="Llama2-Chat-7B", metadata={"help": "Model name to load the right embeddings!"})
    path_prefix: str = field(default="/nfs/ainlp/milang/", metadata={"help": "Path to the root/project directory."})
    pca_dimension: int = field(default=64, metadata={"help": "Size of final/reduced embeddings."})


def train():
    parser = HfArgumentParser(PCAArguments)
    args, = parser.parse_args_into_dataclasses()
    print(args)

    config_name = f"{args.model_name}_{args.dataset}-PCA-{args.pca_dimension}"
    train_data = torch.load(
        f"{args.path_prefix}dresd_files/random_samples/{args.model_name}_{args.dataset}_EMB"
    ).to(dtype=torch.float16)
    print("Loaded embeddings...")
    train_mean = train_data.mean(dim=-2, keepdim=True)
    train_std = train_data.std(dim=-2, keepdim=True)

    # normalize tensors
    train_data = (train_data - train_mean) / train_std
    print("Training PCA...")
    pca_model = PCA(n_components=int(args.pca_dimension), svd_solver="covariance_eigh")
    pca_model = pca_model.fit(train_data)

    torch.save({"pca_model": pca_model, "train_mean": train_mean, "train_std": train_std},
               f"{args.path_prefix}dresd_files/pca_models/{config_name}")
    print(f"{config_name} saved!")

    explained_variance_ratio = pca_model.explained_variance_ratio_
    accum_explained_variance_ratio = torch.cumsum(explained_variance_ratio, dim=-1)
    print(f"Explained Accumulated Variance Ratio: \n{accum_explained_variance_ratio.tolist()}")


if __name__ == '__main__':
    set_seed(42)
    train()
