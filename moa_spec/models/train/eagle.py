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
from typing import Union, Tuple

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class EAGLELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self,
                 config,
                 training_noise,
                 noise_uniform_scale,
                 default_layer,
                 number_of_layers,
                 drafter_num_key_value_heads,
                 drafter_intermediate_size
        ):
        super().__init__(config)

        self.training_noise = training_noise
        self.noise_uniform_scale = noise_uniform_scale
        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")
        self.default_layer = default_layer
        self.number_of_layers = number_of_layers

        if self.default_layer:
            self.spec_model = LlamaDecoderLayer(config, 0)
        else:
            config2 = copy.deepcopy(config)
            config2._attn_implementation = "sdpa"

            config2.num_key_value_heads = drafter_num_key_value_heads
            config2.intermediate_size = drafter_intermediate_size

            self.spec_model = nn.ModuleList([LlamaDecoderLayer(config2, i) for i in range(self.number_of_layers)])

        self.spec_model_fc = nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size, bias=False)

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

    def forward(
        self,
        base_model_output=None,
        input_ids=None,
        attention_mask=None,
        response_mask=None,
        staircase_mask=None,
        epoch=None,
        max_epochs=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, seq_len = input_ids.shape
        token_emb = base_model_output[0].hidden_states[0]
        device = input_ids.device
        response_mask = response_mask[:, 1:].view(-1) if batch_size == 1 else response_mask[:, 1:].reshape(-1)
        response_mask_sum = response_mask.sum() + 1e-5

        X = last_hidden_state = base_model_output[0].hidden_states[-1]
        del base_model_output[0]

        dtype = X.dtype
        min_dtype = torch.finfo(dtype).min

        if self.training_noise:
            X += (torch.rand_like(X) - 0.5) * self.noise_uniform_scale
            token_emb += (torch.rand_like(token_emb) - 0.5) * self.noise_uniform_scale

        attention_mask2 = torch.full(
            (seq_len, seq_len),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        attention_mask2 = torch.triu(attention_mask2, diagonal=1)
        attention_mask2 = attention_mask2[None, None, :, :]

        position_ids = torch.arange(seq_len, device=device)[None]

        Y = torch.cat((token_emb[:, 1:], torch.zeros_like(token_emb[:, 0:1])), 1)
        X2 = torch.cat((Y, X), -1)
        del Y

        X2 = self.spec_model_fc(X2)

        if self.default_layer:
            (attn_output,) = self.spec_model(
                hidden_states=X2,
                attention_mask=attention_mask2,
                position_ids=position_ids,
                use_cache=False,
            )
        else:
            attn_output = X2
            for i in range(self.number_of_layers):
                (attn_output,) = self.spec_model[i](
                    hidden_states=attn_output,
                    attention_mask=attention_mask2,
                    position_ids=position_ids,
                    use_cache=False,
                )

        logits = self.lm_head(attn_output[:, :-1])
        stats = {}
        loss = 0

        # forward KL loss
        plogq = (self.lm_head(last_hidden_state[:, 1:]).softmax(-1) * logits.log_softmax(-1)).sum(-1).view(-1)
        plogq.masked_fill_(torch.isinf(plogq), 0)

        forward_KL = - torch.sum(plogq * response_mask, -1) / response_mask_sum
        stats["forward_KL"] = forward_KL.detach()
        loss += forward_KL * 0.1

        # feature loss
        feature_loss = self.smooth_l1(attn_output[:, :-1], last_hidden_state[:, 1:]).mean(-1).view(-1)
        feature_loss = torch.sum(feature_loss * response_mask, -1) / response_mask_sum
        stats["feature_loss"] = feature_loss.detach()

        loss += feature_loss

        return loss, stats

    def save_pretrained(self, *args, **kwargs):
        state_dict = self.state_dict()
        state_dict = {n: p for n, p in state_dict.items() if "spec" in n}
        kwargs["state_dict"] = state_dict

        return super().save_pretrained(*args, **kwargs)