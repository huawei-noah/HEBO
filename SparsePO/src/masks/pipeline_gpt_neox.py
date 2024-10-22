# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_fx_proxy,
    logging,
)
from typing import Optional, Tuple, Union
from .hooks import (
    AttentionHooks,
    MLPHooks,
    NormalizationHooks,
    TransformerHooks,
)

from .modeling_outputs import CausalLMOutputWithPastMasked
from .modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXPreTrainedModel,
    GPTNeoXModel,
)


class PipelineWrapNL(GPTNeoXForCausalLM):
    _no_split_modules = ["GPTNeoXLayer", "MaskLayer"]
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config, rw_mask=None, kl_mask=None, common=None, mask_config={}):

        GPTNeoXPreTrainedModel.__init__(self,config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rw_mask = rw_mask
        self.kl_mask = kl_mask
        self.common_mask = common
        self.mask_config = mask_config

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hooks: TransformerHooks = TransformerHooks(),
        loss_mask: Optional[torch.FloatTensor] = None,
        reference_states: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastMasked]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        run_mask = reference_states is not None

        outputs = self.gpt_neox(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    hooks=hooks,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)
        if hooks is not None:
            lm_logits = hooks.logits(lm_logits)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        ## mask operations
        rw_mask_out, kl_mask_out = None, None
        mask_on_reward = self.mask_config.get('mask_on_reward',False)
        mask_on_kl = self.mask_config.get('mask_on_kl',False)
        is_rw_kl_independent = self.mask_config.get('is_rw_kl_independent',False)
        mask_model = self.mask_config.get('mask_model','simple_all')
        if run_mask:
            if mask_on_reward:
                mask = self.rw_mask if is_rw_kl_independent else self.common_mask
                if mask_model.endswith("_last"):
                    rw_mask_out = mask(reference_states[-1][:, :-1, :], loss_mask).squeeze(-1)  # (1, d, 1) * (b, seq, d) -> (b, seq, 1)
                elif mask_model.endswith("_all"):
                    rw_mask_out = mask([x[:, :-1, :] for x in reference_states[1:]], loss_mask).squeeze(-1)  # (1, d, 1) * (b, seq, d) -> (b, seq, 1)
                elif mask_model == "random":
                    rw_mask_out = mask.get_mask(reference_states[-1][:, :-1, :]).squeeze(-1)  # (1, d, 1) * (b, seq, d) -> (b, seq, 1)
            if mask_on_kl:
                if not is_rw_kl_independent and mask_on_reward:
                    kl_mask_out = rw_mask_out
                else:
                    mask = self.kl_mask if is_rw_kl_independent else self.common_mask
                    if mask_model.endswith("_last"):
                        kl_mask_out = mask(reference_states[-1][:, :-1, :], loss_mask).squeeze(-1)  # (1, d, 1) * (b, seq, d) -> (b, seq, 1)
                    elif mask_model.endswith("_all"):
                        kl_mask_out = mask([x[:, :-1, :] for x in reference_states[1:]], loss_mask).squeeze(-1)  # (1, d, 1) * (b, seq, d) -> (b, seq, 1)
                    elif mask_model == "random":
                        kl_mask_out = mask.get_mask(reference_states[-1][:, :-1, :]).squeeze(-1)
                #
        #

        if not return_dict:
            output = (lm_logits,) + (rw_mask_out,kl_mask_out) + transformer_outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPastMasked(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                reward_mask=rw_mask_out,
                kl_mask=kl_mask_out,
        )
