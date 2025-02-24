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
import hydra
from typing import Tuple, Union

import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

def create_class(baseclass):
    class Independent(baseclass):
        def __init__(self,
                     config,
                     drafter,
            ):
            super().__init__(config)

            self.spec_model = None
            self.drafter_kwargs = drafter
            self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

        def custom_load(self, load_path):
            self.spec_model = AutoModelForCausalLM.from_pretrained(
                **hydra.utils.instantiate(self.drafter_kwargs),
                device_map=next(self.parameters()).device,
            )
            print(f"LOADING successfully {load_path} states")

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
            _, seq_len = input_ids.shape
            device = attention_mask.device
            position_ids = attention_mask.long().cumsum(-1) - 1

            last_hidden_state = base_model_output[0].hidden_states[-1]
            input_embs = base_model_output[0].hidden_states[0]
            dtype = last_hidden_state.dtype
            min_dtype = torch.finfo(dtype).min
            del base_model_output[0]

            response_mask = response_mask.view(-1)
            response_mask_sum = response_mask.sum() + 1e-5

            causal_mask = torch.full(
                (seq_len, seq_len),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            causal_mask.triu_(diagonal=1)
            causal_mask = causal_mask[None, None]

            spec_model_base = self.spec_model.model(
                inputs_embeds=input_embs,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            attn_output = spec_model_base['last_hidden_state']

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

            loss += feature_loss

            return loss, stats

        def save_pretrained(self, *args, **kwargs):
            return self.spec_model.save_pretrained(*args, **kwargs)

    return Independent

IndependentLlamaForCausalLM = create_class(LlamaForCausalLM)
IndependentQwen2ForCausalLM = create_class(Qwen2ForCausalLM)
