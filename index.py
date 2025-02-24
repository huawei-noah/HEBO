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
import math
import os
import timeit
from dataclasses import dataclass, field
import scann
import torch
import shutil
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoTokenizer
import draftretriever
from datasets import load_from_disk


@dataclass
class IndexArguments:
    dataset: str = field(default="EvolInstructCode", metadata={"help": "Dataset for building the datastore."})
    datastore: str = field(default="DReSD", metadata={"help": "Which datastore? REST or DReSD?"})
    model_name: str = field(default="Llama2-Chat-7B", metadata={"help": "Model used to embed tokens."})
    path_prefix: str = field(default="/nfs/ainlp/milang/", metadata={"help": "Path to the root/project directory."})
    prompt_key: str = field(default="instruction", metadata={"help": "The key to use to get prompt."})
    response_key: str = field(default="output", metadata={"help": "The key to use to get the response."})
    max_length: int = field(default=1024, metadata={"help": "Max sequence length."})
    pca_dimension: int = field(default=64, metadata={"help": "Size of final/reduced embeddings."})


parser = HfArgumentParser(IndexArguments)
args, = parser.parse_args_into_dataclasses()
print(args)
assert args.datastore in ['REST', "DReSD"], "The only choices are 'REST' and 'DReSD'!"

if args.datastore == "REST":
    # ----------------------- BUILD REST -----------------------
    tokenizer = AutoTokenizer.from_pretrained(f"{args.path_prefix}models/{args.model_name}")
    datastore_path = f"{args.path_prefix}dresd_files/{args.model_name}_{args.dataset}_rest_datastore.idx"
    writer = draftretriever.Writer(
        index_file_path=datastore_path,
        max_chunk_len=512*1024*1024,
        vocab_size=tokenizer.vocab_size,
    )
    dataset = load_from_disk(f'{args.path_prefix}datasets/{args.dataset}')
    for ex in tqdm(dataset, total=len(dataset)):
        if type(ex[args.prompt_key]) == list:  # ID datastore
            instruction, output = ex['instruction'],  ex['output']
        else:  # OOD datastore
            instruction = tokenizer.encode(f"Question: {ex[args.prompt_key].strip()}\nAnswer:\n")
            output = tokenizer.encode(f"{ex[args.response_key].strip()}")
        if len(instruction) + len(output) > args.max_length:
            new_len = args.max_length - len(instruction)
            if new_len >= 0:
                response = output[:new_len]
                print(f"Truncating response to {new_len} tokens!")
            else:
                prompt = instruction[:new_len]
                print(f"Truncating prompt to {len(prompt)} tokens!")
        writer.add_entry(instruction + output)

    writer.finalize()

if args.datastore == "DReSD":
    # ----------------------- BUILD SCANN -----------------------
    num_neighbors = 50  # number of neighbours to retrieve
    index_batch_size = 256
    source_path = f"{args.path_prefix}dresd_files/embeddings/{args.model_name}_{args.dataset}_{args.pca_dimension}"
    index_path = f"{args.path_prefix}dresd_files/indices/{args.model_name}_{args.dataset}_{args.pca_dimension}/"
    pca_model = f"{args.path_prefix}dresd_files/pca_models/{args.model_name}_{args.dataset}-PCA-{args.pca_dimension}"

    os.makedirs(index_path, exist_ok=True)
    representations = torch.nn.functional.normalize(torch.load(f"{source_path}_EMB").to(torch.float16), dim=-1).numpy()

    print(f"Starting SCANN indexing from representations of size {representations.shape}.")
    start_time = timeit.default_timer()

    if os.path.exists(os.path.join(index_path, 'index')):
        index = scann.scann_ops_pybind.load_searcher(os.path.join(index_path, 'index'))
        print(f"Loading saved index from '{os.path.join(index_path, 'index')}' ...")
    else:
        print(f"Building an index, then saving to '{os.path.join(index_path, 'index')}' ...")
        index = (
            scann.scann_ops_pybind.builder(representations, num_neighbors, "dot_product")
            .tree(
                num_leaves=int(math.sqrt(len(representations))) * 5,
                num_leaves_to_search=250,
                training_sample_size=int(math.sqrt(len(representations)) * 250),
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(250)
            .build()
        )
        os.makedirs(f"{os.path.join(index_path, 'index')}", exist_ok=True)
        index.serialize(os.path.join(index_path, 'index'))

    # real metadata here!
    shutil.copyfile(f"{source_path}_META", os.path.join(index_path, 'METADATA'))
    shutil.copyfile(pca_model, os.path.join(index_path, 'pca_model'))

    elapsed = timeit.default_timer() - start_time
    print(f"Time for build/load: {elapsed:.3f}")

    top_k_rr = torch.zeros(len(representations))
    start_time = timeit.default_timer()
    bar = tqdm(range(math.ceil(len(representations) / index_batch_size)))

    # run sanity check/evaluation
    for batch_idx in bar:
        start = batch_idx * index_batch_size
        end = min((batch_idx + 1) * index_batch_size, len(representations))
        neighbours, similarities = index.search_batched_parallel(representations[start:end])
        for i in range(start, end):
            try:
                rr = 1 / (1 + list(neighbours[i - start]).index(i))
            except ValueError:
                rr = 0.0
            top_k_rr[i] = rr
        bar.set_description(f"Last batch top K MRR: {top_k_rr[start:end].mean():.2f}")

    elapsed = timeit.default_timer() - start_time
    print(f"Time elapsed for search: {elapsed:.3f}")
    print(f"Overall MRR: {top_k_rr.mean():.3f}")
