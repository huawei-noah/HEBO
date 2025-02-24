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
from datasets import load_from_disk, Dataset
from transformers import set_seed, AutoModelForCausalLM, HfArgumentParser
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class GenArguments:
	local_rank: int = field(default=0, metadata={"help": "Deepspeed will handle/use this."})
	n_gpus: int = field(default=1, metadata={"help": "How many gpus are used for DS Tensor Parallel."})
	dataset: str = field(default="EvolInstructCode", metadata={"help": "Dataset for generation."})
	prompt_key: str = field(default="instruction", metadata={"help": "The key to use to get prompt."})
	model_name: str = field(default="Llama2-Chat-7B", metadata={"help": "Model to use for generation."})
	max_tokens: int = field(default=128, metadata={"help": "Max tokens to generate."})
	max_length: int = field(default=1024, metadata={"help": "Max length includes 'input_ids' and 'max_tokens'."})
	path_prefix: str = field(default="/nfs/ainlp/milang/", metadata={"help": "Path to the root/project directory."})


def main():
	set_seed(42)
	parser = HfArgumentParser(GenArguments)
	args, = parser.parse_args_into_dataclasses()
	print(args)

	data_path = f"{args.path_prefix}datasets/{args.dataset}"
	big_path = f"{args.path_prefix}models/{args.model_name}"
	save_path = f"{args.path_prefix}dresd_files/{args.model_name}_{args.dataset + '_Outputs'}"

	tokenizer = AutoTokenizer.from_pretrained(big_path)
	big = AutoModelForCausalLM.from_pretrained(big_path)
	print(f"Instantiating LM: {big_path}")
	big = deepspeed.init_inference(model=big, dtype=torch.float16, tensor_parallel={"tp_size": args.n_gpus}).module

	big.eval()
	tokenizer.pad_token_id = tokenizer.eos_token_id
	data = load_from_disk(data_path)
	token_data = {"output": [], "instruction": []}

	for i, example in enumerate(tqdm(data)):

		with torch.no_grad():
			prompt = f"Question: {example[args.prompt_key].strip()}\nAnswer:\n"
			inputs = tokenizer.encode(prompt, return_tensors='pt')

			if inputs.shape[1] + args.max_tokens > args.max_length:
				print("WARNING: 'max_length' is too small!")
				continue

			output = big.generate(
				input_ids=inputs.to(big.device),
				max_new_tokens=args.max_tokens,
				do_sample=False,
				# top_p=0.95,
				# temperature=0.7,
				eos_token_id=[tokenizer.eos_token_id],
				pad_token_id=tokenizer.eos_token_id,
			)

			completion = output[:, inputs.shape[1]:]
			print(f"\nINSTRUCTION: {tokenizer.decode(inputs[0].tolist())}")
			print(f"\nOUTPUT: {tokenizer.decode(completion[0].tolist())}")
			token_data['instruction'].extend(inputs.tolist())
			token_data['output'].extend(completion.tolist())

			if i % 2500 == 0 and i > 0:
				Dataset.from_dict(token_data).save_to_disk(save_path)  # overwrite periodically

	Dataset.from_dict(token_data).save_to_disk(save_path)  # save the full dataset at the end as well


if __name__ == '__main__':
	main()
