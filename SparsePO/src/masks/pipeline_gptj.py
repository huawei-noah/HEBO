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

from .modeling_outputs import CausalLMOutputWithPastMasked
from .modeling_gptj import (
    GPTJForCausalLM,
    GPTJPreTrainedModel,
    GPTJModel,
    _REAL_CHECKPOINT_FOR_DOC, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC,GPTJ_INPUTS_DOCSTRING,
)
from typing import Optional, Tuple, Union
from .hooks import (
    AttentionHooks,
    MLPHooks,
    NormalizationHooks,
    TransformerHooks,
)

logger = logging.get_logger(__name__)


class PipelineWrapNLGPTJ(GPTJForCausalLM):
    _no_split_modules = ["GPTJBlock","SimpleMaskLayer"]
    _tied_weights_keys = ["lm_head.weight"]
    """mask_conf: {
        mask_model: str
        mask_on_reward: bool
        mask_on_kl: bool
        is_rw_kl_independent: bool
    }
    """
    def __init__(self, config, rw_mask=None, kl_mask=None, common=None, mask_config={}):
        
        GPTJPreTrainedModel.__init__(self, config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.rw_mask = rw_mask
        self.kl_mask = kl_mask
        self.common_mask = common
        self.mask_config = mask_config

        # Initialize weights and apply final processing
        self.post_init()

        
    # @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=CausalLMOutputWithPastMasked,
    #     config_class=_CONFIG_FOR_DOC,
    #     real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hooks: TransformerHooks = TransformerHooks(),
        loss_mask: Optional[torch.FloatTensor] = None,
        reference_states: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastMasked]:
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        run_mask = reference_states is not None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            hooks=hooks,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        if hooks is not None:
            lm_logits = hooks.logits(lm_logits)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)
        
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
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPastMasked(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            reward_mask=rw_mask_out,
            kl_mask=kl_mask_out,
        )
