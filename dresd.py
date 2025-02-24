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
from typing import Union
import scann
import torch
import draftretriever
from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput


class RestSearch:
	"""REST (exact string suffix search)"""
	def __init__(self, args, eos_id):
		data_path = os.path.join(args.path_prefix, "dresd_files", f"{args.model_name}_{args.datastore}_rest_datastore.idx")
		self.datastore = draftretriever.Reader(index_file_path=data_path)
		self.eos_id = eos_id
		self.args = args

	def search(self, tokens, choices=64):
		token_list = []
		for token_span in range(choices, 2, -1):
			this_token = tokens.squeeze(0)[-token_span:].tolist()
			token_list, _, _, _, _ = self.datastore.search(this_token, choices=choices)
			if len(token_list) > 0:
				new_list = []
				for tl in token_list[:self.args.num_drafts]:
					if -2 in tl:
						tl[tl.index(-2):] = [self.eos_id] * (len(tl) - tl.index(-2))
					new_list.extend([tl])
				token_list = new_list
				break
		if len(token_list) == 0:
			return [[100] for _ in range(self.args.num_drafts)]
		return token_list


class DReSD:
	"""Dense Retrieval for Speculative Decoding"""
	def __init__(self, args, device, pca_dim):
		self.pca_model = torch.load(os.path.join(
			args.path_prefix, "dresd_files", "indices", f"{args.model_name}_{args.datastore}_{args.pca_dimension}", "pca_model")
		)
		self.pca_model["pca_model"].components_ = self.pca_model["pca_model"].components_[:pca_dim, :].to(device=device).to(dtype=torch.float16)
		self.pca_model["pca_model"].mean_ = self.pca_model["pca_model"].mean_.to(device=device).to(dtype=torch.float16)
		self.pca_model["train_mean"] = self.pca_model["train_mean"].to(device=device).unsqueeze(dim=0)
		self.pca_model["train_std"] = self.pca_model["train_std"].to(device=device).unsqueeze(dim=0)
		self.metadata = torch.load(open(os.path.join(
			args.path_prefix, "dresd_files", "indices", f"{args.model_name}_{args.datastore}_{args.pca_dimension}", "METADATA"
		), 'rb'))
		self.index = scann.scann_ops_pybind.load_searcher(os.path.join(
			args.path_prefix, "dresd_files", "indices", f"{args.model_name}_{args.datastore}_{args.pca_dimension}", "index")
		)
		self.generation_config = args  # for HF compatibility
		self.generation_config.num_assistant_tokens = args.len_drafts
		self.device = device

	@torch.no_grad()
	def generate(self, big_embedding=None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:

		input_ids = kwargs["input_ids"]
		if big_embedding is None or kwargs['max_new_tokens'] == 1:
			pass
		else:
			norm_query = (big_embedding[None, None] - self.pca_model["train_mean"]) / self.pca_model["train_std"]
			norm_query = self.pca_model["pca_model"].transform(norm_query.squeeze(0))
			norm_query = torch.nn.functional.normalize(norm_query, dim=-1)
			neighbours, similarities = self.index.search(norm_query[0].cpu().numpy())
			first_token = input_ids[0][-1].cpu()  # we will filter candidates/neighbours by first token
			next_token_ids = [self.metadata[i][:kwargs['max_new_tokens']] for i in neighbours if self.metadata[i][0] == first_token]
			if len(next_token_ids) > 0:
				next_token_ids = torch.tensor(next_token_ids, device=self.device, dtype=torch.int64)[:self.generation_config.num_drafts, 1:]
				input_ids = torch.cat([input_ids.repeat(next_token_ids.size(0), 1), next_token_ids], dim=-1)

		return GenerateDecoderOnlyOutput(
			sequences=input_ids,
			scores=None,
			logits=None,
			hidden_states=None,
			past_key_values=None,
		)


class REST:

	def __init__(self, args, device, eos_id):
		self.index = RestSearch(args, eos_id)
		self.generation_config = args  # for HF compatibility
		self.generation_config.num_assistant_tokens = args.len_drafts
		self.device = device

	@torch.no_grad()
	def generate(self, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:

		input_ids = kwargs["input_ids"]
		next_token_ids = [l[:kwargs['max_new_tokens']] for l in self.index.search(input_ids)]
		next_token_ids = torch.tensor(next_token_ids, device=input_ids.device)
		input_ids = torch.cat(
			[
				input_ids.repeat(self.generation_config.num_drafts, 1),
				next_token_ids.view(self.generation_config.num_drafts, -1)
			],
			dim=-1
		)

		return GenerateDecoderOnlyOutput(
			sequences=input_ids,
			scores=None,
			logits=None,
			hidden_states=None,
			past_key_values=None,
		)
