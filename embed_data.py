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
import os
from dataclasses import dataclass, field
import torch
import deepspeed
from datasets import load_from_disk
from transformers import set_seed, AutoModelForCausalLM, HfArgumentParser
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EmbedArguments:
	local_rank: int = field(default=0, metadata={"help": "Deepspeed will handle/use this."})
	n_gpus: int = field(default=1, metadata={"help": "How many cards are used for DS Tensor Parallel."})
	dataset: str = field(default="EvolInstructCode", metadata={"help": "Dataset for the datastore."})
	prompt_key: str = field(default="instruction", metadata={"help": "The key to use to get prompt."})
	response_key: str = field(default="output", metadata={"help": "The key to use to get the response."})
	model_name: str = field(default="Llama2-Chat-7B", metadata={"help": "Model to use to embed tokens."})
	max_length: int = field(default=1024, metadata={"help": "Max sequence length."})
	only_dump_big_emb: bool = field(default=False, metadata={"help": "Only dump training embeddings for PCA."})
	pca_dimension: int = field(default=64, metadata={"help": "Size of final/reduced embeddings."})
	path_prefix: str = field(default="/nfs/ainlp/milang/", metadata={"help": "Path to the root/project directory."})
	n_len: int = field(default=20, metadata={"help": "The length of the draft next (N) tokens."})


def main():
	set_seed(42)
	parser = HfArgumentParser(EmbedArguments)
	args, = parser.parse_args_into_dataclasses()
	print(args)

	data_path = f"{args.path_prefix}datasets/{args.dataset}"
	model_path = f"{args.path_prefix}models/{args.model_name}"
	folder_name = "random_samples" if args.only_dump_big_emb else "embeddings"
	dimension = "" if args.only_dump_big_emb else f"_{args.pca_dimension}"
	save_path = f"{args.path_prefix}dresd_files/{folder_name}/{args.model_name}_{args.dataset}{dimension}"

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path)
	print(f"Instantiated LM: {model_path}")
	model = deepspeed.init_inference(model, dtype=torch.float16, tensor_parallel={"tp_size": args.n_gpus}).module

	if args.only_dump_big_emb:
		pca_model = None
		tensors = torch.randn((1, model.config.hidden_size))
	else:
		pca_model = torch.load(f"{args.path_prefix}dresd_files/pca_models/{args.model_name}_{args.dataset}-PCA-{args.pca_dimension}")
		pca_model["pca_model"].components_ = pca_model["pca_model"].components_[:args.pca_dimension, :].to(model.device)
		pca_model["pca_model"].mean_ = pca_model["pca_model"].mean_.to(model.device)
		pca_model["train_mean"] = pca_model["train_mean"].to(model.device).unsqueeze(dim=0)
		pca_model["train_std"] = pca_model["train_std"].to(model.device).unsqueeze(dim=0)
		tensors = torch.randn((1, pca_model["pca_model"].components_.size(0)))

	model.eval()
	padding = [tokenizer.eos_token_id] * args.n_len
	in_data = load_from_disk(data_path).shuffle(seed=42)
	in_data = in_data.flatten_indices()  # for 10x speed

	meta_data = {'next_k_tokens': []}
	tmp_tensors = []  # to speed up execution loop
	for i, example in enumerate(tqdm(in_data)):

		if args.only_dump_big_emb and tensors.size(0) > 1_000_000:
			break  # exit after ~1M tokens to train PCA

		with torch.inference_mode():
			if type(example[args.prompt_key]) == list:
				prompt = example[args.prompt_key]
				response = example[args.response_key]
			else:
				prompt = tokenizer.encode(f"Question: {example[args.prompt_key].strip()}\nAnswer:\n")
				response = tokenizer.encode(f"{example[args.response_key.strip()]}")
			if len(prompt) + len(response) > args.max_length:
				new_len = args.max_length - len(prompt)
				if new_len >= 0:
					response = response[:new_len]
					print(f"Truncating response to {new_len} tokens!")
				else:
					prompt = prompt[:new_len]
					print(f"Truncating prompt to {len(prompt)} tokens!")

			inputs = torch.tensor([prompt + response])
			outputs = model(inputs.to(model.device), output_hidden_states=True)
			if args.only_dump_big_emb:
				hidden_states = outputs.hidden_states[-1].tolist()
			else:
				hidden_states = (outputs.hidden_states[-1] - pca_model["train_mean"]) / pca_model["train_std"]
				hidden_states = pca_model["pca_model"].transform(hidden_states).to(dtype=torch.float16).tolist()
				meta_data['next_k_tokens'].extend([inputs[0, i + 1: i + 1 + args.n_len].tolist() + padding[inputs.shape[1] - 1 - i:]
					                               for i in range(inputs.shape[1] - 1)])

			tmp_tensors.extend(hidden_states[0][:-1])
			if len(tmp_tensors) > 100_000:
				tensors = torch.cat([tensors, torch.tensor(tmp_tensors, dtype=torch.float16)], dim=0)
				tmp_tensors = []

	if len(tmp_tensors) > 0:
		tensors = torch.cat([tensors, torch.tensor(tmp_tensors, dtype=torch.float16)], dim=0)
	# save tensors for indexing
	torch.save(tensors[1:, :], f"{save_path}_EMB")
	print(f"Embedding Size: {tensors[1:, :].shape}")

	if not args.only_dump_big_emb:
		# save metadata for retrieval
		assert all([len(md) == args.n_len for md in meta_data['next_k_tokens']])
		meta_iterable = [[i for i in range(len(meta_data['next_k_tokens']))], meta_data['next_k_tokens']]
		meta_dict = {k: v1 for k, v1 in zip(*meta_iterable)}
		assert tensors[1:, :].size(0) == len(meta_data['next_k_tokens'])
		torch.save(meta_dict, f"{save_path}_META")


if __name__ == '__main__':
	main()
