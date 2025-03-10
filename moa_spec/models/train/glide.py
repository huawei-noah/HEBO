# MIT License
#
# Copyright (c) 2024, Huawei Technologies Co., Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import os
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from typing import Tuple, Union

import torch
from safetensors import safe_open
from torch import nn
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer

from moa_spec.models.train.cross_attention import GlideCrossAttention


def create_class(baseclass, baselayer):
    class Glide(baseclass):
        def __init__(self,
                     config,
                     self_attention_num_key_value_heads,
                     self_attention_intermediate_size,
                     block_size=5,
            ):
            super().__init__(config)
            config2 = copy.deepcopy(config)
            self.block_size = block_size

            self.cross_attention = GlideCrossAttention(config2, 0)

            config2._attn_implementation = "sdpa"
            config2.num_key_value_heads = self_attention_num_key_value_heads
            config2.intermediate_size = self_attention_intermediate_size
            self.self_attention = baselayer(config2, 0)

        # modeling_lama forgot LlamaRMSNorm during initialization
        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, LlamaRMSNorm):
                module.weight.data.normal_(mean=0.0, std=std)
            elif isinstance(module, Qwen2RMSNorm):
                module.weight.data.normal_(mean=0.0, std=std)

        def forward(
            self,
            base_model_output=None,
            input_ids=None,
            attention_mask=None,
            response_mask=None,
            epoch=None,
            max_epochs=None,
            **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

            stats = {}
            batch_size, seq_len = input_ids.shape
            device = attention_mask.device
            position_ids = attention_mask.long().cumsum(-1) - 1

            # Keep only useful information from base_model_output and clean memory
            last_hidden_state = base_model_output[0].hidden_states[-1]
            input_embs = base_model_output[0].hidden_states[0]
            dtype = last_hidden_state.dtype
            min_dtype = torch.finfo(dtype).min

            KV = base_model_output[0]['past_key_values'][-1]
            del base_model_output[0]
            # torch.cuda.empty_cache() # slow down but fit more memory

            response_mask = response_mask.view(-1)
            response_mask_sum = response_mask.sum() + 1e-5

            # Self-Attention
            attention_mask_sa = torch.full(
                (seq_len, seq_len),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            attention_mask_sa.triu_(diagonal=1)
            attention_mask_sa = attention_mask_sa[None, None]

            # Decompose self_attention call to place cross_attention in between
            # as done in the original Glide repository

            # (attn_output,) = self.self_attention(
            #     hidden_states=input_embs,
            #     attention_mask=attention_mask_sa,
            #     position_ids=position_ids,
            #     use_cache=False,
            # )

            residual = input_embs
            hidden_states = self.self_attention.input_layernorm(input_embs)

            # Self Attention
            hidden_states, _, _ = self.self_attention.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask_sa,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states

            # Cross Attention

            # Glide's block mask, simplified for batch_size == 1
            assert batch_size == 1
            block_indices = torch.arange(seq_len, device=device) // self.block_size
            attention_condition = block_indices.unsqueeze(1) < block_indices.unsqueeze(0) + 1
            attention_mask_ca = attention_condition.to(dtype=dtype)
            attention_mask_ca[attention_mask_ca.eq(1)] = min_dtype
            attention_mask_ca = attention_mask_ca[None, None]

            (hidden_states, _, _) = self.cross_attention(
                hidden_states=hidden_states,
                hidden_states2=KV,
                attention_mask=attention_mask_ca,
                position_ids=position_ids,
                use_cache=False,
            )

            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.self_attention.post_attention_layernorm(hidden_states)
            hidden_states = self.self_attention.mlp(hidden_states)
            hidden_states = residual + hidden_states

            logits = self.lm_head(hidden_states)
            loss = 0

            # forward KL loss
            plogq = (self.lm_head(last_hidden_state).detach().softmax(-1) * logits.log_softmax(-1)).sum(-1).view(-1)
            plogq.masked_fill_(torch.isinf(plogq), 0)

            forward_KL = - torch.sum(plogq * response_mask, -1) / response_mask_sum
            stats["forward_KL"] = forward_KL.detach()
            loss += forward_KL * 0.1

            return loss, stats

        def save_pretrained(self, *args, **kwargs):
            state_dict = self.state_dict()
            state_dict = {n: p for n, p in state_dict.items()
                          if ("self_attention" in n or
                              "cross_attention" in n or
                              "layer_self_attention" in n or
                              "Ekv2E" in n)
            }
            kwargs["state_dict"] = state_dict

            return super().save_pretrained(*args, **kwargs)

        def custom_load(self, load_path):
            state_dict = {}
            if os.path.exists(os.path.join(load_path, "model.safetensors")):
                with safe_open(os.path.join(load_path, "model.safetensors"), framework="pt", device=self.device.type) as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                raise()

            if state_dict == {}:
                print("WARNING : failed to load previous LLM state")
            else:
                print(f"LOADING successfully {load_path} states ({len(state_dict)} numbers of keys)")
            self.load_state_dict(state_dict, strict=False)

    return Glide

GlideLlamaForCausalLM = create_class(LlamaForCausalLM, LlamaDecoderLayer)
GlideQwen2ForCausalLM = create_class(Qwen2ForCausalLM, Qwen2DecoderLayer)
