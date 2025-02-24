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
import time
import deepspeed
import numpy as np
from dataclasses import dataclass, field
import torch
from itertools import chain
from datasets import load_from_disk, Dataset
from transformers import set_seed, AutoModelForCausalLM, HfArgumentParser
from transformers import AutoTokenizer
from tqdm import tqdm
from dresd import DReSD, REST
from spec_dec_dresd import SpecDecDReSD
from spec_dec_base import SpecDecBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvaluateArguments:
	local_rank: int = field(default=0, metadata={"help": "Deepspeed will handle/use this."})
	n_gpus: int = field(default=1, metadata={"help": "How many cards are used for DS Tensor Parallel."})
	dresd: bool = field(default=False, metadata={"help": "Use the DReSD model."})
	sd: str = field(default=None, metadata={"help": "Baseline SD, options: 'small' or 'pld'."})
	rest: bool = field(default=False, metadata={"help": "Whether to use REST (suffix matching)?"})
	small_name: str = field(default="Llama-Chat-68M", metadata={"help": "Whether to use v2 (model path) or not (None)?"})
	dataset: str = field(default="CodeAlpaca", metadata={"help": "Dataset for evaluation."})
	datastore: str = field(default="EvolInstructCode", metadata={"help": "Datastore name."})
	prompt_key: str = field(default="instruction", metadata={"help": "The key to use to get prompt."})
	model_name: str = field(default="Llama2-Chat-7B", metadata={"help": "Model(s) to evaluate."})
	max_tokens: int = field(default=128, metadata={"help": "Max tokens to generate."})
	num_drafts: int = field(default=10, metadata={"help": "How many nearest neighbours to use as drafts."})
	len_drafts: int = field(default=10, metadata={"help": "How (many tokens) long should each draft be?"})
	max_length: int = field(default=2048, metadata={"help": "Max length includes 'input_ids' and 'max_tokens'."})
	pca_dimension: int = field(default=64, metadata={"help": "Size of final/reduced embeddings."})
	path_prefix: str = field(default="/nfs/ainlp/milang/", metadata={"help": "Path to the root/project directory."})


def main():
	set_seed(42)
	parser = HfArgumentParser(EvaluateArguments)
	args, = parser.parse_args_into_dataclasses()
	print(args)

	data_path = f"{args.path_prefix}datasets/{args.dataset}"
	model_path = f"{args.path_prefix}models/{args.model_name}"
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path)
	model = deepspeed.init_inference(model, dtype=torch.float16, tensor_parallel={"tp_size": args.n_gpus}).module

	if args.dresd:
		model.generation_config.output_hidden_states = True
		args.len_drafts = args.len_drafts + 1  # +1 for the 'filtering token', see methodology (Section 3)
		dresd = DReSD(args=args, device=model.device, pca_dim=64)
		model = SpecDecDReSD(model, dresd)
		print(f"Instantiating DReSD with LM:\n{model_path}")
	elif args.rest:
		rest = REST(args, model.device, tokenizer.eos_token_id)
		model = SpecDecBase(model, rest)
		print(f"Instantiating REST with LM:\n{model_path}")
	elif args.sd is not None and args.sd == "pld":
		model.generation_config.prompt_lookup_num_tokens = args.len_drafts
		model = SpecDecBase(model, None)
		print(f"Instantiating Prompt Lookup Decoding...")
	elif args.sd is not None and args.sd == "small":
		assistant = AutoModelForCausalLM.from_pretrained(f"{args.path_prefix}models/{args.small_name}")
		assistant = deepspeed.init_inference(assistant, dtype=torch.float16, tensor_parallel={"tp_size": args.n_gpus}).module
		assistant.generation_config.num_assistant_tokens = args.len_drafts
		assistant.generation_config.num_drafts = args.num_drafts
		model = SpecDecBase(model, assistant)
		print(f"Instantiating Assistant LM - {args.small_name}")
	else:
		print(f"Instantiating Baseline LM - {args.model_name}")

	model.eval()
	tokenizer.pad_token_id = tokenizer.eos_token_id
	data = load_from_disk(data_path)
	if args.dataset == "CodeAlpaca":
		data = data.shuffle(seed=42).select(range(100))
	if args.dataset == "MT-Bench":
		data = Dataset.from_list([{args.prompt_key: d['turns'][0]} for d in data])

	metrics = []
	start = time.time()
	for i, example in enumerate(tqdm(data)):

		with torch.no_grad():
			prompt = f"Question: {example[args.prompt_key].strip()}\nAnswer:\n"
			inputs = tokenizer.encode(prompt, return_tensors='pt')

			if inputs.shape[1] + args.max_tokens > args.max_length:
				print("WARNING: 'max_length' is too small!")
				continue

			output = model.generate(
				input_ids=inputs.to(model.device),
				max_new_tokens=args.max_tokens,
				do_sample=False,
				# top_p=0.95,
				# temperature=0.7,
				eos_token_id=[tokenizer.eos_token_id],
				pad_token_id=tokenizer.eos_token_id,
			)

			if args.dresd or args.sd:
				output, output_metrics = output
			else:
				output, output_metrics = output, {}

			completion = output[0][inputs.shape[1]:]
			output_metrics["total_tokens"] = len(completion)
			metrics.append(output_metrics)

	end = time.time()
	print(f'Generation time: {end - start:.2f} seconds.')
	print(f'Tokens-per-second: {sum(m["total_tokens"] for m in metrics) // int(end - start)}')
	print(f'Mean Tokens Generated: {sum(m["total_tokens"] for m in metrics) / len(metrics)}')
	if args.dresd or args.sd:
		gen_t, acc_t = [m["generated_tokens"] for m in metrics], [m["accepted_tokens"] for m in metrics]
		print(f'Acceptance Rate (overall): {sum(chain(*acc_t)) / sum(chain(*gen_t)):.3f}')
		print(f'Acceptance Rate (sequence): {(list(chain(*acc_t)) / np.array(list(chain(*gen_t)))).mean():.3f}')
		print(f'Mean Drafted Length: {np.array(list(chain(*gen_t))).mean():.2f}')
		print(f'Mean Accepted Length: {np.array(list(chain(*acc_t))).mean():.2f}')
		print(f'Mean Calls to BIG: {np.array([len(m["generated_tokens"]) for m in metrics]).mean():.1f}')


if __name__ == '__main__':
	main()
