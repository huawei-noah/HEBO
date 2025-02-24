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

from moa_spec.models.train.cross_attention import LlamaDecoderLayerCrossAttention


def create_class(baseclass, baselayer):
    class MOASpec(baseclass):
        def __init__(self,
                     config,
                     layer_self_attention_num_key_value_heads,
                     layer_self_attention_intermediate_size,
                     self_attention_num_key_value_heads,
                     self_attention_intermediate_size,
                     cross_attention_num_key_value_heads,
                     cross_attention_intermediate_size,
                     target_layer_inference,
                     variable_future_length,
                     staircase_mask
            ):
            super().__init__(config)
            self.target_layer_inference = target_layer_inference
            self.variable_future_length = variable_future_length
            self.staircase_mask = staircase_mask
            self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

            config2 = copy.deepcopy(config)
            config2._attn_implementation = "sdpa"

            config2.num_key_value_heads = self_attention_num_key_value_heads
            config2.intermediate_size = self_attention_intermediate_size
            self.self_attention = baselayer(config2, 0)

            config2.num_key_value_heads = cross_attention_num_key_value_heads
            config2.intermediate_size = cross_attention_intermediate_size
            self.cross_attention = LlamaDecoderLayerCrossAttention(config2, 0)

            config2.hidden_size = 2 * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
            config2.num_key_value_heads = layer_self_attention_num_key_value_heads
            config2.intermediate_size = layer_self_attention_intermediate_size
            config2.head_dim = config2.hidden_size // config2.num_attention_heads
            self.layer_self_attention = baselayer(config2, 0)

            self.Ekv2E = nn.Linear(config2.hidden_size, config.hidden_size)

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
            target_layer_inf_input = base_model_output[0].hidden_states[- 1 - self.target_layer_inference]
            last_hidden_state = base_model_output[0].hidden_states[-1]
            input_embs = base_model_output[0].hidden_states[0]
            nb_layers = len(base_model_output[0]['past_key_values'])
            dtype = last_hidden_state.dtype
            min_dtype = torch.finfo(dtype).min

            KV = torch.stack([
                torch.cat(kv, dim=-1).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
                for kv in base_model_output[0]['past_key_values']], 2)
            del base_model_output[0]
            # torch.cuda.empty_cache() # slow down but fit more memory

            if self.variable_future_length:
                response_mask_noisy = response_mask.clone()
                every = torch.randint(low=6, high=15, size=(batch_size,), device=device)
                indices = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
                every_expanded = every.unsqueeze(1).expand(batch_size, seq_len)
                mask = (indices % every_expanded) == 0
                response_mask_noisy[mask] = 0
            else:
                response_mask_noisy = response_mask

            response_mask = response_mask.view(-1)
            response_mask_sum = response_mask.sum() + 1e-5

            staircase_mask = None
            if self.staircase_mask:
                probability = epoch / (max_epochs / 2) if epoch < max_epochs / 2 else 1.0
                random_values = torch.rand(size=(batch_size,), device=device)
                staircase_mask = ~(random_values < probability)

                stats["staircase_mask"] = staircase_mask.float().mean()

            # Layer Self-Attention
            KV = KV.reshape(-1, nb_layers, KV.shape[-1])
            (KV, ) = self.layer_self_attention(
                hidden_states=KV,
                attention_mask=torch.zeros((1, 1, nb_layers, nb_layers), device=device, dtype=dtype),
                position_ids=torch.arange(nb_layers, device=device, dtype=position_ids.dtype)[None],
                use_cache=False,
            )
            KV = KV.mean(1).reshape(batch_size, seq_len, -1)
            KV = self.Ekv2E(KV)

            # Self-Attention
            seq_len_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            start_positions_candidate = torch.where(
                response_mask_noisy == 0,
                seq_len_indices,
                torch.tensor(0, device=device)
            )
            start_positions, _ = torch.cummax(start_positions_candidate, dim=1)  # Shape: (batch_size, seq_len)

            idx_i = seq_len_indices.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
            idx_j = seq_len_indices.unsqueeze(1)  # Shape: (batch_size, 1, seq_len)

            start_positions_expanded = start_positions.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

            if staircase_mask is not None and staircase_mask.any():
                attention_mask_sa = torch.full(
                    (batch_size, seq_len, seq_len),
                    fill_value=min_dtype,
                    dtype=dtype,
                    device=device,
                )

                mask_3d = (idx_j >= start_positions_expanded + 1) & (idx_j <= idx_i) | (idx_j == idx_i)
                attention_mask_sa[mask_3d] = 0.0

                for i in range(batch_size):
                    if not staircase_mask[i]:
                        attention_mask_sa[i].triu_(diagonal=1)

                attention_mask_sa.unsqueeze_(1)
            else:
                attention_mask_sa = torch.full(
                    (seq_len, seq_len),
                    fill_value=min_dtype,
                    dtype=dtype,
                    device=device,
                )
                attention_mask_sa.triu_(diagonal=1)
                attention_mask_sa = attention_mask_sa[None, None]

            (attn_output,) = self.self_attention(
                hidden_states=input_embs,
                attention_mask=attention_mask_sa,
                position_ids=position_ids,
                use_cache=False,
            )

            # Cross Attention
            attention_mask_ca = torch.full(
                (batch_size, seq_len, seq_len),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )

            mask_3d = (idx_j <= start_positions_expanded) & (idx_j <= idx_i)
            attention_mask_ca[mask_3d] = 0.0
            attention_mask_ca.unsqueeze_(1)

            (attn_output,) = self.cross_attention(
                hidden_states=attn_output,
                hidden_states2=KV,
                attention_mask=attention_mask_ca,
                position_ids=position_ids,
                use_cache=False,
            )

            # Target Layer Inference
            if self.target_layer_inference > 0:
                attention_mask_tli = torch.full(
                    (seq_len, seq_len),
                    fill_value=min_dtype,
                    dtype=dtype,
                    device=device,
                )
                attention_mask_tli.triu_(diagonal=1)
                attention_mask_tli = attention_mask_tli[None, None]

                # no need for a deepcopy here, the loss will be defined only on the non-modified part
                target_layer_inf_input_pred = attn_output
                # assume errors because fully on-policy needs multiple forward pass
                mask = (response_mask_noisy == 0).unsqueeze(-1).expand_as(attn_output)
                attn_output[mask] = target_layer_inf_input[mask]

                for i in range(self.target_layer_inference, 0, -1):
                    (attn_output,) = self.model.layers[-i](
                        hidden_states=attn_output,
                        attention_mask=attention_mask_tli,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                attn_output = self.model.norm(attn_output)

                prev_feature_loss = self.smooth_l1(target_layer_inf_input_pred, target_layer_inf_input).mean(-1).view(-1)
                prev_feature_loss = torch.sum(prev_feature_loss * response_mask, -1) / response_mask_sum
                stats["prev_feature_loss"] = prev_feature_loss.detach()

            logits = self.lm_head(attn_output)
            loss = 0

            # forward KL loss
            plogq = (self.lm_head(last_hidden_state).detach().softmax(-1) * logits.log_softmax(-1)).sum(-1).view(-1)
            plogq.masked_fill_(torch.isinf(plogq), 0)

            forward_KL = - torch.sum(plogq * response_mask, -1) / response_mask_sum
            stats["forward_KL"] = forward_KL.detach()
            loss += forward_KL * 0.1

            # feature loss
            feature_loss = self.smooth_l1(attn_output, last_hidden_state).mean(-1).view(-1)
            feature_loss = torch.sum(feature_loss * response_mask, -1) / response_mask_sum
            stats["feature_loss"] = feature_loss.detach()

            # additional feature loss on previous layers when target layer is set
            if self.target_layer_inference:
                feature_loss = feature_loss * 0.5 + prev_feature_loss * 0.5

            loss += feature_loss

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

    return MOASpec

MOASpecLlamaForCausalLM = create_class(LlamaForCausalLM, LlamaDecoderLayer)
MOASpecQwen2ForCausalLM = create_class(Qwen2ForCausalLM, Qwen2DecoderLayer)
